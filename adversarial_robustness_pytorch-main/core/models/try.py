from model_zoo import WideResNet
a= WideResNet()
print(a)
#for param_tensor in a.state_dict():
#    print(param_tensor, "\t", a.state_dict()[param_tensor].size())
