def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
