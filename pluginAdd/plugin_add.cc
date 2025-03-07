// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "plugin_add.h" // NOLINT

#include <plugin/register_op.h>

#include <memory>
#include <string>
#include <vector>

namespace TECO_INFER
{

  struct plugin_addAttrs : public tvm::AttrsNode<plugin_addAttrs>
  {
    double alpha;
    std::string beta_str;
    TVM_DECLARE_ATTRS(plugin_addAttrs, "relay.attrs.plugin_addAttrs")
    {
      TVM_ATTR_FIELD(alpha).set_default(1.0).describe("Alpha attribute");
      TVM_ATTR_FIELD(beta_str).set_default("").describe("Beta attribute");
    }
  };
  TVM_REGISTER_NODE_TYPE(plugin_addAttrs);

  class PluginAddImpl : public AbstractPluginOp
  {
  public:
    PluginAddImpl() = default;

    void InferOutputShape(const std::vector<std::vector<int>> &total_input_shape, int n_input,
                          std::vector<std::vector<int>> &total_output_shape, int n_output,
                          const OpAttr &attr)
    {
      float alpha = attr.GetAttr<float>("alpha");
      std::cout << "alpha = " << alpha << std::endl;

      std::string beta_str = attr.GetAttr<std::string>("beta_str");
      std::cout << "beta_str = " << beta_str << std::endl;
      total_output_shape[0] = total_input_shape[0];
    }

    void Enqueue(std::shared_ptr<ComputeContext> &ctx)
    {
      std::cout << "CALL PluginAddd enqueue" << std::endl;
      void *input1_dev = ctx->GetInputDataPtr("input1");
      void *input2_dev = ctx->GetInputDataPtr("input2");
      void *output_dev = ctx->GetOutputDataPtr(0);

      float alpha;
      ctx->GetAttr("alpha", alpha);

      std::vector<int> input1_shape;
      ctx->GetInputShape("input1", input1_shape);

      int total_elements = 1;
      for (int i : input1_shape)
      {
        total_elements *= i;
      }

      sdaaStream_t stream = ctx->GetStream();

      PluginAddForward(input1_dev, input2_dev, output_dev, alpha, total_elements, stream, DATA_FLOAT);
    }
  };

  REGISTER_PLUGIN_OP_IMPL(plugin_add, PluginAddImpl)

  class PluginAddImplSymbol : public SymbolPluginOp
  {
  public:
    void SymbolInferOutput(std::vector<ISymShape> &total_input_shapes,
                           std::vector<ISymShape> &total_output_shapes, const OpAttr &attr)
    {
      total_output_shapes[0] = total_input_shapes[0];
    }
  };

  REGISTER_PLUGIN_SYMBOLIC_OP(plugin_add, PluginAddImplSymbol)

  PLUGIN_REGISTER_OP("plugin_add")
      .Input("input1")
      .Type("Tensor")
      .Desc("Left operator")
      .Input("input2")
      .Type("Tensor")
      .Desc("Right operator")
      .AttrType<plugin_addAttrs>()
      .Register();
} // namespace TECO_INFER
