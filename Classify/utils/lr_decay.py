def param_groups_lrd_geomoe(model, base_lr, weight_decay=1e-5, layer_decay=0.75):
    block_len = [len(model.blocks1), len(model.blocks2), len(model.blocks3)]
    num_layers = sum(block_len) + 4 + 1
    layer_scales = [layer_decay**(num_layers - i - 1) for i in range(num_layers)]
    param_groups = {}

    for n, p in model.named_parameters():

        if n.startswith("blocks1"):
            layer = int(n.split(".")[1]) + 1
        elif n.startswith("blocks2"):
            layer = int(n.split(".")[1]) + 2 + block_len[0]
        elif n.startswith("blocks3"):
            layer = int(n.split(".")[1]) + 4 + block_len[0] + block_len[1]
        elif n.startswith("patch_embed1") or n.startswith("pos_embed"):
            layer = 0
        elif n.startswith("patch_embed2"):
            layer = 1 + block_len[0]
        elif n.startswith("patch_embed3"):
            layer = 2 + block_len[0] + block_len[1]
        elif n.startswith("patch_embed4"):
            layer = 3 + block_len[0] + block_len[1]
        elif n.startswith("stage"):
            layer = num_layers - 2
        else:
            layer = num_layers - 1
        
        if p.ndim == 1:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        group_name = f"layer_{layer}_{group_name}"
        if group_name not in param_groups:
            param_groups[group_name] = {
                "params": [],
                "weight_decay": this_weight_decay,
                "lr": layer_scales[layer] * base_lr
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def param_groups_lrd_moe(model, base_lr, weight_decay=1e-5, layer_decay=0.75):
    block_len = len(model.blocks)
    num_layers = block_len + 1
    layer_scales = [layer_decay**(num_layers - i - 1) for i in range(num_layers)]
    param_groups = {}

    for n, p in model.named_parameters():

        if n.startswith("blocks"):
            layer = int(n.split(".")[1]) + 1
        elif n.startswith("patch_embed") or n.startswith("pos_embed") or n.startswith("cls_token"):
            layer = 0
        else:
            layer = num_layers - 1
        
        if p.ndim == 1:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        group_name = f"layer_{layer}_{group_name}"
        if group_name not in param_groups:
            param_groups[group_name] = {
                "params": [],
                "weight_decay": this_weight_decay,
                "lr": layer_scales[layer] * base_lr
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())

moe_lrd = param_groups_lrd_moe
geo_lrd = param_groups_lrd_geomoe