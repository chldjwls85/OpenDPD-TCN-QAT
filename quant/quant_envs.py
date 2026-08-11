import torch
import torch.nn as nn
import copy

from .modules.gru import GRU as PYGRU
from .modules.ops import Mul, Add, Sqrt, Pow
from .qmodules.quantizers import (
    DISCARD_LSB_SIGNED_FLOOR,
    ROUND_TO_NEAREST_TIES_TO_EVEN,
    Identity_Quantizer, INT_Quantizer, OP_INT_Quantizer
    )
from .qmodules.quant_layers import INT_Conv1D, INT_Conv2D, INT_Linear, INT_Pass
from .qmodules.quant_activations import INT_Hardswish
from .qmodules.quant_ops import Quant_sigmoid, Quant_tanh, Quant_mult, Quant_add, Quant_sqrt, Quant_pow

# Modified in the OpenDPD-TCN-QAT fork to add a dedicated Conv1d QAT path.


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value


class InputOutputQuantWrapper(nn.Module):
    """Quantize raw and predistorted I/Q at the physical interface grid."""

    def __init__(self, model, bits):
        super().__init__()
        if int(bits) < 2:
            raise ValueError("full-I/O signed quantization requires at least 2 bits")
        self.model = model
        self.backbone_type = model.backbone_type
        self.input_quantizer = INT_Quantizer(bits, all_positive=False)
        self.output_quantizer = INT_Quantizer(bits, all_positive=False)
        self.input_quantizer.init_act_params()
        self.output_quantizer.init_act_params()
        self._boundary_bits = int(bits)
        boundary_scale = 2.0 ** (1 - self._boundary_bits)
        for quantizer in (self.input_quantizer, self.output_quantizer):
            # The physical ADC/DAC grid is an interface contract, not a learned
            # QAT parameter.  Keeping it as a persistent buffer excludes it from
            # every optimizer while preserving the checkpoint key.
            quantizer._parameters.pop("scale")
            quantizer.register_buffer(
                "scale", torch.tensor([boundary_scale], dtype=torch.float32)
            )
        self.assert_physical_io_scales()

    def assert_physical_io_scales(self):
        """Require the exact signed-unit physical interface scale."""
        expected = 2.0 ** (1 - self._boundary_bits)
        for name, quantizer in (
            ("raw_input", self.input_quantizer),
            ("dpd_output", self.output_quantizer),
        ):
            value = float(quantizer.scale.detach().item())
            if value != expected:
                raise ValueError(
                    f"{name} physical scale must be exactly 2^(1-A)="
                    f"{expected}, got {value}"
                )
            if isinstance(quantizer.scale, torch.nn.Parameter):
                raise TypeError(f"{name} physical scale must be a non-trainable buffer")
        return expected

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, x, h_0=None):
        quantized_input = self.input_quantizer(x)
        output = self.model(quantized_input, h_0)
        return self.output_quantizer(output)


def create_quantizer(type, n_bits, all_positive, act_or_weight):
    quantizer_types = ['INT_Quantizer', 'Identity_Quantizer',
                       'Drf_Act_Quantizer', 'Drf_Weight_Quantizer',
                       'IAO_Quantizer',
                       'FP8_Quantizer',
                       'PACT_Quantizer',
                    ]
    assert type in quantizer_types, 'Quantizer type {} is not supported.'.format(type)
    if 'INT_Quantizer' in type:
        quantizer = INT_Quantizer(n_bits, all_positive)
    elif 'Identity_Quantizer' in type:
        quantizer = Identity_Quantizer(n_bits, all_positive)
    else:
        raise NotImplementedError('Quantizer type {} is not implemented.'.format(type))
    return quantizer

def recur_rpls_layers(args, model, layer_type=nn.Conv2d,
                      rpls_layer_type=INT_Conv2D,
                      weight_quantizer=INT_Quantizer(8, all_positive=False),
                      act_quantizer=INT_Quantizer(8, all_positive=False)):
    """ Recursively replace layers of a given type with another type within a model.
    Args:
        model: the model to be searched.
        layer_type: the type of the layer to be replaced.
        rpls_layer_type: the type of the layer to be replaced with.
    Returns:
        A list of layers of the given type.
    """

    for name, module in model.named_children():
        weight_quantizer = create_quantizer(weight_quantizer.__class__.__name__, args.n_bits_w, all_positive=False, act_or_weight='weight')
        act_quantizer = create_quantizer(act_quantizer.__class__.__name__, args.n_bits_a, all_positive=False, act_or_weight='act')
        if isinstance(module, layer_type):
            print('Replace {} with {}'.format(layer_type, rpls_layer_type))
            setattr(model, name, rpls_layer_type(module, weight_quantizer, act_quantizer))
        else:
            recur_rpls_layers(args, module, layer_type, rpls_layer_type, weight_quantizer, act_quantizer)


def recur_rpls_hardswish(model, bits, rounding):
    """Replace every hardware-path HardSwish with an input-quantized module."""

    for name, module in model.named_children():
        if isinstance(module, nn.Hardswish):
            setattr(model, name, INT_Hardswish(bits=bits, rounding=rounding))
        else:
            recur_rpls_hardswish(module, bits, rounding)

def create_op_quantizer(type, n_bits, all_positive):
    quantizer_types = ['OP_INT_Quantizer', 'Identity_Quantizer', 'Drf_Act_Quantizer', 'IAO_Quantizer', 'FP8_Quantizer']
    assert type in quantizer_types, 'Quantizer type {} is not supported.'.format(type)
    if 'OP_INT_Quantizer' in type:
        quantizer = OP_INT_Quantizer(n_bits, all_positive)
    elif 'Identity_Quantizer' in type:
        quantizer = Identity_Quantizer(n_bits, all_positive)
    else:
        raise NotImplementedError('Quantizer type {} is not implemented.'.format(type))
    return quantizer

def recur_rpls_ops(args, model, op_type, rpls_op_type, *quantizers):
    """ Recursively replace layers of a given type with another type within a model.
    Args:
        model: the model to be searched.
        layer_type: the type of the layer to be replaced.
        rpls_layer_type: the type of the layer to be replaced with.
    Returns:
        A list of layers of the given type.
    """
    sigmoid_quantizer, tanh_quantizer, mult_quantizer, add_quantizer, \
    sqrt_quantizer, pow_quantizer = quantizers

    for name, module in model.named_children():
        if isinstance(module, op_type):
            # print('Replace {} with {}'.format(op_type, rpls_op_type))
            if isinstance(module, torch.nn.Sigmoid):
                sigmoid_quantizer = create_op_quantizer(sigmoid_quantizer.__class__.__name__, sigmoid_quantizer.bits, sigmoid_quantizer.all_positive)
                setattr(model, name, rpls_op_type(sigmoid_quantizer))
            elif isinstance(module, torch.nn.Tanh):
                tanh_quantizer = create_op_quantizer(tanh_quantizer.__class__.__name__, tanh_quantizer.bits, tanh_quantizer.all_positive)
                setattr(model, name, rpls_op_type(tanh_quantizer))
            elif isinstance(module, Mul):
                mult_quantizer = create_op_quantizer(mult_quantizer.__class__.__name__, mult_quantizer.bits, mult_quantizer.all_positive)
                setattr(model, name, rpls_op_type(mult_quantizer))
            elif isinstance(module, Add):
                add_quantizer = create_op_quantizer(add_quantizer.__class__.__name__, add_quantizer.bits, add_quantizer.all_positive)
                setattr(model, name, rpls_op_type(add_quantizer))
            elif isinstance(module, Sqrt):
                sqrt_quantizer = create_op_quantizer(sqrt_quantizer.__class__.__name__, sqrt_quantizer.bits, sqrt_quantizer.all_positive)
                setattr(model, name, rpls_op_type(sqrt_quantizer))
            elif isinstance(module, Pow):
                pow_quantizer = create_op_quantizer(pow_quantizer.__class__.__name__, pow_quantizer.bits, pow_quantizer.all_positive)
                setattr(model, name, rpls_op_type(module, pow_quantizer))
            else:
                raise NotImplementedError('Operation type {} is not implemented.'.format(op_type))
            # print("model: ", model)
        else:
            recur_rpls_ops(args, module, op_type, rpls_op_type, *quantizers)


def recur_rpls_gru(model):
    """ Recursively replace GRU module with the self-defined pytorch GRU module.
    Args:
        model: the model to be searched.
    Returns:
        A list of layers of the given type.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.GRU):
            setattr(model, name, PYGRU(input_size = module.input_size,
                                       hidden_size = module.hidden_size,
                                       num_layers = module.num_layers,
                                       batch_first = module.batch_first,
                                       bias=module.bias is not None)
                    )
        else:
            recur_rpls_gru(module)


class FExLiteTCNQuantEnv:
    """Native full-I/O QAT environment for ``fexlite_causal_tcn``."""

    def __init__(self, model, args):
        self.args = args
        self.model = copy.deepcopy(model)
        self.n_bits_w = int(args.n_bits_w)
        self.n_bits_a = int(args.n_bits_a)
        self.quantize_hardswish_input = bool(
            getattr(args, "quantize_hardswish_input", False)
        )
        self.activation_rounding = getattr(
            args, "activation_rounding", ROUND_TO_NEAREST_TIES_TO_EVEN
        )
        if self.activation_rounding not in {
            ROUND_TO_NEAREST_TIES_TO_EVEN,
            DISCARD_LSB_SIGNED_FLOOR,
        }:
            raise ValueError(
                f"unsupported TCN activation rounding: {self.activation_rounding}"
            )
        if self.n_bits_w < 2 or self.n_bits_a < 2:
            raise ValueError("signed TCN QAT requires activation and weight bits >= 2")
        if getattr(args, "pretrained_model", ""):
            state = torch.load(
                args.pretrained_model, map_location="cpu", weights_only=True
            )
            incompatible = self.model.load_state_dict(state, strict=False)
            allowed_missing = {"backbone._rtl_spec"}
            unexpected_missing = set(incompatible.missing_keys) - allowed_missing
            if unexpected_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "pretrained TCN checkpoint is incompatible: "
                    f"missing={sorted(unexpected_missing)}, "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )

        quantized = copy.deepcopy(self.model)
        recur_rpls_layers(
            args,
            quantized,
            nn.Conv1d,
            INT_Conv1D,
            INT_Quantizer(self.n_bits_w, all_positive=False),
            INT_Quantizer(self.n_bits_a, all_positive=False),
        )
        conv_layers = [
            module for module in quantized.modules()
            if isinstance(module, INT_Conv1D)
        ]
        # Conv0 consumes FEx output and keeps the existing RNE contract.  Every
        # later convolution consumes a post-HardSwish activation, so its input
        # fake quantizer is the explicit post-activation boundary.
        for layer in conv_layers[1:]:
            layer.act_quantizer.rounding = self.activation_rounding

        if self.quantize_hardswish_input:
            recur_rpls_hardswish(
                quantized, self.n_bits_a, self.activation_rounding
            )

        if (
            self.quantize_hardswish_input
            or self.activation_rounding != ROUND_TO_NEAREST_TIES_TO_EVEN
        ):
            rounding_code = (
                1 if self.activation_rounding == DISCARD_LSB_SIGNED_FLOOR else 0
            )
            quantized.backbone.register_buffer(
                "_activation_quant_spec",
                torch.tensor(
                    [1, int(self.quantize_hardswish_input), rounding_code,
                     self.n_bits_a],
                    dtype=torch.int64,
                ),
            )
        self.q_model = InputOutputQuantWrapper(quantized, self.n_bits_a)

    @staticmethod
    def _covering_power_of_two_scale(clip, qmax):
        required = clip.clamp_min(1e-12) / qmax
        return torch.pow(
            required.new_tensor(2.0), torch.ceil(torch.log2(required))
        )

    @torch.no_grad()
    def calibrate(self, loader, device):
        """Calibrate each Conv1d input in graph order using training data only."""
        quantile = float(self.args.quant_calibration_quantile)
        max_batches = int(self.args.quant_calibration_batches)
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quant_calibration_quantile must be in (0, 1]")
        if max_batches < 1:
            raise ValueError("quant_calibration_batches must be positive")

        calibration_points = []
        conv_index = 0
        hardswish_index = 0
        for module in self.q_model.modules():
            if isinstance(module, INT_Conv1D):
                calibration_points.append((
                    f"conv{conv_index}_input", module.act_quantizer, False
                ))
                conv_index += 1
            elif isinstance(module, INT_Hardswish):
                calibration_points.append((
                    f"hardswish{hardswish_index}_input",
                    module.input_quantizer,
                    True,
                ))
                hardswish_index += 1
        calibration_batches = []
        for batch_index, (features, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            calibration_batches.append(features.detach().cpu())
        if not calibration_batches:
            raise ValueError("calibration loader produced no batches")
        was_training = self.q_model.training
        self.q_model.eval()
        result = {
            "raw_input": {
                "scale": float(self.q_model.input_quantizer.scale.item()),
                "bits": self.n_bits_a,
                "policy": "fixed_signed_unit_interface",
            },
            "dpd_output": {
                "scale": float(self.q_model.output_quantizer.scale.item()),
                "bits": self.n_bits_a,
                "policy": "fixed_signed_unit_interface",
            },
        }
        try:
            for label, quantizer, must_represent_threshold in calibration_points:
                clip = torch.zeros((), device=device)

                def capture(_module, inputs):
                    nonlocal clip
                    values = inputs[0].detach().abs().flatten()
                    clip = torch.maximum(clip, torch.quantile(values, quantile))

                handle = quantizer.register_forward_pre_hook(capture)
                try:
                    for features in calibration_batches:
                        self.q_model(features.to(device))
                finally:
                    handle.remove()
                design_clip = torch.maximum(
                    clip,
                    clip.new_tensor(3.0) if must_represent_threshold
                    else clip.new_tensor(0.0),
                )
                scale = self._covering_power_of_two_scale(
                    design_clip, quantizer.Qp
                )
                quantizer.scale.copy_(scale)
                result[label] = {
                    "clip": float(clip.cpu()),
                    "design_clip": float(design_clip.cpu()),
                    "scale": float(scale.cpu()),
                    "bits": self.n_bits_a,
                    "rounding": quantizer.rounding,
                    "quantile": quantile,
                    "batches": len(calibration_batches),
                }
        finally:
            self.q_model.train(was_training)
        return result

class Base_GRUQuantEnv(object):
    """ Base class for quantization environment
    Args:
        args: arguments
        model: the model to be quantized.
    """
    def __init__(self, model, args=AttrDict()):
        self.args = args
        self.model = model

        self.n_bits_w = args.n_bits_w
        self.n_bits_a = args.n_bits_a

        self.fq_layers_hash = {
            nn.Conv2d: INT_Conv2D,
            nn.Linear: INT_Linear,
        }
        self.fq_ops_hash = {
            nn.Sigmoid: Quant_sigmoid,
            nn.Tanh: Quant_tanh,
            Mul: Quant_mult,
            Add: Quant_add,
            Sqrt: Quant_sqrt,
            Pow: Quant_pow,
        }

        self.last_layer_type = INT_Pass

        # quantizers
        self.weight_quantizer, self.act_quantizer,  \
        self.sigmod_quantizer, self.tanh_quantizer, \
        self.mult_quantizer, self.add_quantizer,    \
        self.sqrt_quantizer, self.pow_quantizer        = self.set_quantizer()

        # float model
        self.pygru_model = self.create_pygru_model(copy.deepcopy(self.model))
        self.pygru_model = self.load_model(self.pygru_model)

        # quantized model
        self.q_model = self.create_quantized_model(copy.deepcopy(self.pygru_model))

    def load_model(self, model):
        pretrained_model = self.args.pretrained_model
        use_pretrained = bool(pretrained_model)

        if use_pretrained:
            model.load_state_dict(torch.load(pretrained_model))
            print("Load pretrained model from {}".format(pretrained_model))
        else:
            print("No pretrained model is loaded.")
        return model

    def set_quantizer(self):
        print('INT Quantizers are used.')
        weight_quantizer = INT_Quantizer(self.n_bits_w, all_positive=False)
        act_quantizer = INT_Quantizer(self.n_bits_a, all_positive=False)
        # weight_quantizer = IAO_Quantizer(bits=self.n_bits_w, all_positive=False, act_or_weight='weight')
        # act_quantizer = IAO_Quantizer(bits=self.n_bits_a, all_positive=False, act_or_weight='act')
        # weight_quantizer = Drf_Weight_Quantizer(bits=self.n_bits_w, all_positive=False)
        # act_quantizer = Drf_Act_Quantizer(bits=self.n_bits_a, all_positive=True)

        # weight_quantizer = FP8_Quantizer(self.n_bits_w, all_positive=False)
        # act_quantizer = Identity_Quantizer(self.n_bits_a, all_positive=False)

        sigmod_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        tanh_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        mult_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)
        add_quantizer = OP_INT_Quantizer(self.n_bits_a, all_positive=False)


        # sigmod_quantizer = Drf_Act_Quantizer(self.n_bits_a, all_positive=False)
        # tanh_quantizer = Drf_Act_Quantizer(self.n_bits_a, all_positive=False)
        # mult_quantizer = Drf_Act_Quantizer(self.n_bits_w, all_positive=False)
        # add_quantizer = Drf_Act_Quantizer(self.n_bits_w, all_positive=False)
        # sqrt_quantizer = OP_INT_Quantizer(bits=16, all_positive=False)
        # pow_quantizer = OP_INT_Quantizer(bits=16, all_positive=False)
        sqrt_quantizer = Identity_Quantizer(self.n_bits_w, all_positive=False)
        pow_quantizer = Identity_Quantizer(self.n_bits_w, all_positive=False)


        return weight_quantizer, act_quantizer, sigmod_quantizer, tanh_quantizer, mult_quantizer, add_quantizer, \
               sqrt_quantizer, pow_quantizer

    def create_pygru_model(self, model):
        """ Create a pytorch GRU model from the original model.
        Args:
            model: the original model.
        Returns:
            A model with pytorch GRU module.
        """
        def _reset_parameters(model, hidden_size):
            for name, param in model.named_parameters():
                num_gates = int(param.shape[0] / hidden_size)
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                if 'weight' in name:
                    for i in range(0, num_gates):
                        nn.init.orthogonal_(param[i * hidden_size:(i + 1) * hidden_size, :])
                if 'x2h.weight' in name:
                    for i in range(0, num_gates):
                        nn.init.xavier_uniform_(param[i * hidden_size:(i + 1) * hidden_size, :])

        def _reset_pygru(model):
            for name, module in model.named_children():
                if isinstance(module, PYGRU):
                    print("::: Reset pytorch GRU module.")
                    _reset_parameters(module, module.hidden_size)
                else:
                    _reset_pygru(module)

        # replace GRU module with pytorch GRU module
        recur_rpls_gru(model)

        # reset parameters
        _reset_pygru(model)

        return model

    def unquantize_last_layer(self, model, last_layer_name='fc_out'):
        """ Unquantize the last layer of the model.
        Args:
            model: the model to be added with the last layer.
            last_layer_name: the name of the last layer.
        Returns:
            A model with the a unquantized last layer.
        """
        for name, module in model.named_children():
            if name == last_layer_name:
                print("Unquantize the last layer: ", name)
                module.weight_quantizer = Identity_Quantizer()
                module.act_quantizer = Identity_Quantizer()
            else:
                self.unquantize_last_layer(module, last_layer_name)

    def set_first_layer(self, model, first_layer_name='x2h'):
        """ Set the first layer attributes of the model.
        """
        for name, module in model.named_children():
            if name == first_layer_name:
                print("Set the first layer: ", name)
                module.act_quantizer.bits = 16
            else:
                self.set_first_layer(module, first_layer_name)

    def set_last_layer_quant(self, model, last_layer_name='fc_out'):
        """ Set the last layer attributes of the model.
        """
        for name, module in model.named_children():
            if name == last_layer_name:
                print("quant the output")
                module.out_quant = True
            else:
                self.set_last_layer_quant(module, last_layer_name)

    def create_quantized_model(self, model):
        """ Create a quantized model from the original model.
        Args:
            model: the original model.
        Returns:
            A quantized pygru model.
        """

        for op_type, rpls_op_type in self.fq_ops_hash.items():
            recur_rpls_ops(self.args, model, op_type, rpls_op_type, \
                self.sigmod_quantizer, self.tanh_quantizer, self.mult_quantizer, self.add_quantizer, \
                self.sqrt_quantizer, self.pow_quantizer)

        for layer_type, rpls_layer_type in self.fq_layers_hash.items():
            recur_rpls_layers(self.args, model, layer_type, rpls_layer_type, self.weight_quantizer, self.act_quantizer)

        # self.set_first_layer(model)
        # self.unquantize_last_layer(model)
        self.set_last_layer_quant(model)

        return model
