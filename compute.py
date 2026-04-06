import torch

def compute_layer_complete_with_restoration(
    layer_idx, inputs_embeds, attention_mask, position_ids, und_expert, gen_expert, act_expert, prefix_router, num_tokens_to_keep
):
    """
    使用临时剪枝和恢复策略计算一个完整的层。

    Args:
        layer_idx (int): 当前层的索引。
        inputs_embeds (list[torch.Tensor]): 包含不同专家输入的列表。
        attention_mask (torch.Tensor): 注意力掩码。
        position_ids (torch.Tensor): 位置ID。
        und_expert: 理解专家模型.
        gen_expert: 生成专家模型.
        act_expert: 行动专家模型.
        prefix_router: 用于为 token 打分的路由器网络。
        num_tokens_to_keep (int): 每层需要保留的 token 数量。

    Returns:
        tuple[list[torch.Tensor], torch.Tensor]: 返回更新后的 embeds 列表和该层的 OT loss。
    """
    models = [und_expert.language_model, gen_expert, act_expert]
    query_states = []
    key_states = []
    value_states = []
    layer_ot_loss = 0.0  # 初始化本层的 OT Loss

    # ---- 新的临时剪枝与恢复逻辑 ----
    # 假设剪枝只作用于第一个输入 (und_expert)
    original_prefix_hidden_states = inputs_embeds[0]
    batch_size, original_sequence_length, hidden_dim = original_prefix_hidden_states.shape

    # 1. 计算 router 分数并获取 Top-K token
    router_output = prefix_router(original_prefix_hidden_states)
    
    # 可以在这里计算 OT Loss (如果需要的话), 沿用你之前的逻辑
    # layer_ot_loss = calculate_ot_loss(...) 
    # 注意：calculate_ot_loss 需要你自己实现，可以参考图片中的逻辑

    scores, indices_to_keep = torch.topk(router_output.squeeze(-1), num_tokens_to_keep, dim=1)
    indices_to_keep, _ = torch.sort(indices_to_keep, dim=1)

    # 2. 创建一个稀疏的 hidden_states 用于计算
    # 创建一个全零的 "画布"
    pruned_hidden_states = torch.zeros_like(original_prefix_hidden_states)
    
    # 将被选中的 token "散布" (scatter) 到画布的相应位置
    # indices_for_gather 的形状需要是 (batch_size, num_tokens_to_keep, hidden_dim)
    indices_for_gather = indices_to_keep.unsqueeze(-1).expand(batch_size, -1, hidden_dim)
    
    # 先用 gather 选出重要的 token
    selected_tokens = torch.gather(original_prefix_hidden_states, 1, indices_for_gather)
    
    # 再用 scatter_ 将这些 token 放回稀疏画布的正确位置
    pruned_hidden_states.scatter_(1, indices_for_gather, selected_tokens)
    
    # 创建处理后的 inputs_embeds 列表
    # 只有 prefix (第一个输入) 被剪枝了
    processed_inputs_embeds = [pruned_hidden_states] + inputs_embeds[1:]

    # ---- 从这里开始，使用 processed_inputs_embeds 进行计算 ----
    # 因为序列长度没有改变，所以 attention_mask 和 position_ids 无需修改

    for i, hidden_states in enumerate(processed_inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)
        
        # --- 接下来的代码与你原始版本中的 Q, K, V 计算部分基本相同 ---
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)

        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    # Concatenate and process attention (这部分与你原始代码一致)
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0], query_states.shape[2], query_states.shape[-1],
        device=query_states.device, dtype=query_states.dtype,
    )
    cos, sin = und_expert.model.language_model.rotary_emb(dummy_tensor, position_ids)
    
    # 注意：这里的 apply_rotary_pos_emb 假设可以处理维度未变的 position_ids
    # 如果 position_ids 在之前被错误地剪枝了，这里需要使用原始的 position_ids
    query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    # Attention computation (与原始代码一致, attention_mask也使用原始的)
    att_output, _ = modeling_qwen3_vl.eager_attention_forward(
        und_expert.language_model.layers[layer_idx].self_attn,
        query_states, key_states, value_states, attention_mask,
        und_expert.language_model.layers[layer_idx].self_attn.scaling,
    )
    
    head_dim = und_expert.language_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_expert.language_model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, 1 * num_attention_heads * head_dim)

    # ---- 恢复与残差连接 ----
    # 这一部分与你的原始代码非常相似，但 now, 它是正确的，
    # 因为 att_output 的维度与原始 inputs_embeds 的维度匹配。
    outputs_embeds = []
    start_pos = 0
    # 在残差连接时，我们需要使用未经剪枝的原始输入
    for i, original_input in enumerate(inputs_embeds): 
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + original_input.shape[1]
        
        current_att_output = att_output[:, start_pos:end_pos]
        if current_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            current_att_output = current_att_output.to(layer.self_attn.o_proj.weight.dtype)
            
        out_emb = layer.self_attn.o_proj(current_att_output)
        
        # First residual: 关键！将计算结果与原始的、未经剪枝的 hidden_states 相加
        # 被剪枝的 token 对应的 att_output 部分为 0，所以它们的特征通过这个残差连接被“恢复”了
        out_emb = out_emb + original_input

        # 后续 MLP 和第二个残差连接与你的原始代码一致
        after_first_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)

        out_emb = layer.mlp(out_emb)
        out_emb = out_emb + after_first_residual
        
        outputs_embeds.append(out_emb)
        start_pos = end_pos
        
    # 返回结果，可以根据需要带上 loss
    return outputs_embeds # 或者 return outputs_embeds, layer_ot_loss