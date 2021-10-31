#Attribution
    #Integrated Gradients
    #Saliency
    #DeepLift
    #DeepLiftShap
    #GradientShap
    #Input X Gradient
    #Guided Backprop
    #Guided GradCAM
    #Deconvolution
    #Feature Ablation
    #Occlusion

#Feature Attributions - Integrated Gradients
ig = IntegratedGradients(net) #For a network with multiple outputs, a target index must also be provided, defining the index of the output for which gradients are computed
test_input_tensor.requires_grad_() #test_input_tensor is the test tensor for which we are calculating this. Grads needed.
attr, delta = ig.attribute(test_input_tensor,target=1, return_convergence_delta=True)
attr = attr.detach().numpy() 
importances = np.mean(attr, axis=0)

#Layer Attributions - LayerConductance - Layer attributions allow us to understand the importance of all the neurons in the output of a particular layer.
cond = LayerConductance(net, net.sigmoid1) 
cond_vals = cond.attribute(test_input_tensor,target=1)
cond_vals = cond_vals.detach().numpy()
importances = np.mean(cond_vals, axis=0)

a = []