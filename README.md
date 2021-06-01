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


# Code Flow:-
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

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite the following in your publications:

> [Soumick Chatterjee, Mario Breitkopf, Chompunuch Sarasaen, Hadya Yassin, Georg Rose, Andreas Nürnberger, Oliver Speck: ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data (arXiv:2103.09203, Mar 2021)](https://arxiv.org/abs/2103.09203)

BibTeX entry:

```bibtex
@article{chatterjee2021reconresnet,
  title={ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data},
  author={Chatterjee, Soumick and Breitkopf, Mario and Sarasaen, Chompunuch and Yassin, Hadya and Rose, Georg and N{\"u}rnberger, Andreas and Speck, Oliver},
  journal={arXiv preprint arXiv:2103.09203},
  year={2021}
}
```

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
