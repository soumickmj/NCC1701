# NCC1701	
Official code of the paper "ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data" (https://arxiv.org/abs/2103.09203).

Initial work was presented at ESMRMB 2019, Rotterdam.
Abstract available on RG: https://www.researchgate.net/publication/335473585_A_deep_learning_approach_for_reconstruction_of_undersampled_Cartesian_and_Radial_data

Extension of this work will be getting updated continously on this repo.

One of the extension has been accepted for presentation at ISMRM 2021.
Abstract available on RG: https://www.researchgate.net/publication/349589092_Going_beyond_the_image_space_undersampled_MRI_reconstruction_directly_in_the_k-space_using_a_complex_valued_residual_neural_network and the talk is available on YouTube: https://www.youtube.com/watch?v=WWMyUtQOrqg

The name of this project "NCC1701" was selected as a tribute to starships "USS Enterprise" from the Star Trek franchise.
Registry number of the first USS Enterprise is NCC-1701. Future publications will be given names with the subsequent registry numbers of ships bearing the name USS Enterprise.

# Off-topic (Maybe!): Little bit about USS Enterprise
(Source: https://memory-alpha.fandom.com/wiki/Enterprise_history)

The USS Enterprise (NCC-1701) "will be" a Constitution-class starship that will serve under five captains from 2245 to 2285, including Robert April, Christopher Pike, Will Decker, and in later years Spock. Its most famous commander will be Captain James T. Kirk, whose five-year mission aboard the Enterprise will become legendary.

There will be five subsequent starships bearing the name USS Enterprise with registry numbers from NCC-1701-A to NCC-1701-E, and will serve under four captains: James T. Kirk, John Harriman, Rachel Garrett, and Jean-Luc Picard (will breifly serve under William T. Riker and Edward Jellico).


## Code Flow
Main -> MRITorchDS (dsClass)
     -> Model (netModel)
     -> Engine (Model) -> Helper
        -> Constructor
		-> CreateModel
		-> IntitializeMRITorchDS
			-> TorchDSInitializer
		-> Train -> self.helpOBJ.save_checkpoint
			-> TrainOneEpoch -> Helper.EvaluationParams, helpOBJ.Domain2Image
		-> TrainNValidate
		-> Test -> self.helpOBJ.kSpace2ImgSpace
     -> load checkpoint
     -> LRDecay
     -> undersampling mask
        ->     

 ## Model Weights
The weights of the models trained on the IXI dataset (results presented in Section 3.2 of the article) have been made publicly available on Huggingface, and they can be found in the collection: [https://huggingface.co/collections/soumickmj/reconresnet-ncc1701-66d32635cf97d8801971138a](https://huggingface.co/collections/soumickmj/reconresnet-ncc1701-66d32635cf97d8801971138a). The designations "IXIT1Guys" and "IXIT1HH" within the model names indicate that the respective model was trained on the IXI T1-weighted MRIs acquired at Guy’s Hospital using a Philips 1.5T system (IXIT1Guys), and at Hammersmith Hospital using a Philips 3T system (IXIT1HH).

An additional set of models, labelled with "IXIT1HHGuys_BET" in their names, was trained on the brain-extracted versions of the T1-weighted MRIs from both Guy’s Hospital (1.5T) and Hammersmith Hospital (3T). These models, while not part of the original article, may prove useful in scenarios where a model compatible with both 1.5T and 3T MRI, or a model designed to work with brain-extracted T1-weighted MRIs, is required. Furthermore, another supplementary set of models, identified by the "DS6MRA" label, was trained on the 7T MRA dataset used in DS6 (https://doi.org/10.3390/jimaging8100259). Although these models are not part of the original article, they could be beneficial for applications requiring models that operate with MRAs, particularly 7T MRAs.

Since these models have been uploaded to Hugging Face following their format, but the NCC1701 pipeline here is incompatible with Hugging Face directly, they cannot be used as-is. Instead, the weights must be downloaded using the AutoModel class from the transformers package, saved as a checkpoint, and then the path to this saved checkpoint must be specified for the executor using the "--preweights_path" parameter.

Here is an example:
```python
from transformers import AutoModel
modelHF = AutoModel.from_pretrained("soumickmj/NCC1701_ReconResNet2D_IXIT1HH_Varden1D15", trust_remote_code=True)
torch.save({'state_dict': modelHF.model.state_dict()}, "/path/to/checkpoint/model.pth")
```
To run the NCC1701 pipeline with these weights, the path to the checkpoint must then be passed as preweights_path, as an additional parameter along with the other desired parameters:
```bash
--preweights_path /path/to/checkpoint/model.pth
```

Although this model was created using PyTorch, the trained weights can also be utilised with other libraries, courtesy of Hugging Face. For further information, please refer to the Transformers documentation on [Huggingface](https://huggingface.co/docs/transformers).

# Contacts

Please feel free to contact me for any questions or feedback:

[soumick.chatterjee@ovgu.de](mailto:soumick.chatterjee@ovgu.de)

[contact@soumick.com](mailto:contact@soumick.com)

# Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite the following in your publications:

> [Soumick Chatterjee, Mario Breitkopf, Chompunuch Sarasaen, Hadya Yassin, Georg Rose, Andreas Nürnberger, Oliver Speck: ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data (Computers in Biology and Medicine, Apr 2022)](https://doi.org/10.1016/j.compbiomed.2022.105321)

BibTeX entry:

```bibtex
@article{chatterjee2021reconresnet,
  title={ReconResNet: Regularised residual learning for MR image reconstruction of Undersampled Cartesian and Radial data},
  author={Chatterjee, Soumick and Breitkopf, Mario and Sarasaen, Chompunuch and Yassin, Hadya and Rose, Georg and N{\"u}rnberger, Andreas and Speck, Oliver},
  journal={Computers in Biology and Medicine},
  pages={105321},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2022.105321}
}
```
The complete manuscript is also on ArXiv:-
> [Soumick Chatterjee, Mario Breitkopf, Chompunuch Sarasaen, Hadya Yassin, Georg Rose, Andreas Nürnberger, Oliver Speck: ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data (arXiv:2103.09203, Mar 2021)](https://arxiv.org/abs/2103.09203)

For the Complex (AKA K-Space) version of the ReconResNet, please (additionally) cite the following:

> [Soumick Chatterjee, Chompunuch Sarasaen, Alessandro Sciarra, Mario Breitkopf, Steffen Oeltze-Jafra, Andreas Nürnberger, Oliver Speck: Going beyond the image space: undersampled MRI reconstruction directly in the k-space using a complex valued residual neural network (ISMRM, May 2021)](https://www.researchgate.net/publication/349589092_Going_beyond_the_image_space_undersampled_MRI_reconstruction_directly_in_the_k-space_using_a_complex_valued_residual_neural_network)

BibTeX entry:

```bibtex
@inproceedings{mickISMRM21ksp,
      author = {Chatterjee, Soumick and Sarasaen, Chompunuch and Sciarra, Alessandro and Breitkopf, Mario and Oeltze-Jafra, Steffen and Nürnberger, Andreas and                     Speck, Oliver},
      year = {2021},
      month = {05},
      pages = {1757},
      title = {Going beyond the image space: undersampled MRI reconstruction directly in the k-space using a complex valued residual neural network},
      booktitle={2021 ISMRM \& SMRT Annual Meeting \& Exhibition}
}
```
Thank you so much for your support.
