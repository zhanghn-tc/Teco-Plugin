# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import onnx
from onnx import helper, TensorProto
import tvm
from tvm import relay
import tvm.relay

from tvm.plugin import plugins
import tecoinference
from tvm.contrib.teco_infer_dyn import dyn
import numpy as np


def create_plugin_add_onnx_model(op_type, input_shapes, output_shape, attributes=None):
    input_names = list(input_shapes.keys())
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in zip(input_names, input_shapes)
    ]
    output_name = 'output'
    output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)

    attributes = attributes or {}
    add_node = helper.make_node(op_type,
                                input_names, [output_name],
                                **attributes,
                                domain="my_custom_ops",
                                version=21)

    graph = helper.make_graph(
        [add_node],
        op_type,
        inputs,
        [output],
    )
    model = helper.make_model(graph)
    opset_imports = model.opset_import
    opset_imports.append(helper.make_opsetid("my_custom_ops", 21))
    return model


def test_plugin_add_exec():
    model_add = create_plugin_add_onnx_model(op_type='plugin_add',
                                             input_shapes={
                                                 "input1": (2, 3),
                                                 "input2": (2, 3)
                                             },
                                             output_shape=(2, 3),
                                             attributes={
                                                 'alpha': 1.2,
                                                 'beta_str': 'beta_abc'
                                             })

    plugins.register_op(op_name="plugin_add",
                        inputs=["input1", "input2"],
                        attrs={
                            "alpha": "float",
                            "beta_str": "string"
                        })
    mod, prams = tvm.relay.frontend.from_onnx(model_add, {"input1": (relay.Any(), relay.Any()), "input2": (relay.Any(), relay.Any())})
    print("plugin_add_ir:", mod)
    
    min_shapes = {"input1": [1, 1], "input2": [1, 1]}
    max_shapes = {"input1": [4, 4], "input2": [4, 4]}
    
    mod = relay.transform.SymbolicConstantFolder(min_shapes, max_shapes)(mod)
    print("symbol ir:", mod)
    fbs_model = dyn.to_teco_infer_dyn(mod, {}, "teco_dyn")
    engine = tecoinference.Engine(fbs_model)
    ctx = engine.create_context()
    input1_data = np.random.randn(2, 3).astype("float32")
    input2_data = np.random.randn(2, 3).astype("float32")

    ctx.set_input(0, input1_data)
    ctx.set_input(1, input2_data)
    ctx.executor_run()
    out = ctx.get_output(0)

    print("input1_data: ", input1_data)
    print("input2_data: ", input2_data)
    print("out: ", out)
    np_out = (input1_data + input2_data) * 1.2
    print("np_out: ", np_out)

    np.testing.assert_allclose(out, np_out)

test_plugin_add_exec()


if __name__ == "__main__":
    pass
