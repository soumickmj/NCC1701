import torch
from torchio.transforms import IntensityTransform, Transform
import torchio as tio

from Engineering.transforms import transforms as cusTrans

class IntensityNorm(IntensityTransform):
    def __init__(
            self,
            type: str = "minmax",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.transformer = cusTrans.IntensityNorm(type=type, applyonly=True)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                transformed_tensors.append(self.transformer(tensor.float()))
            image.set_data(torch.stack(transformed_tensors))
        return subject

class ForceAffine(Transform):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.inp.affine = subject.gt.affine
        return subject

class ChangeDataSpace(IntensityTransform):
    def __init__(
            self,
            source_data_space, 
            destin_data_space,
            data_dim = (-3,-2,-1),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.transformer = cusTrans.ChangeDataSpace(source_data_space=source_data_space, destin_data_space=destin_data_space, data_dim=data_dim, applyonly=True)

    def apply_transform(self, subject: tio.Subject):
        if self.source_data_space == self.destin_data_space:
            return subject
        for name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            image.set_data(image.data.float())
            for tensor in image.data:
                transformed_tensors.append(self.transformer(tensor))
            image.set_data(torch.stack(transformed_tensors))
        return subject

def getDataSpaceTransforms(dataspace_inp, model_dataspace_inp, dataspace_gt, model_dataspace_gt):
    if dataspace_inp == dataspace_gt and model_dataspace_inp == model_dataspace_gt and dataspace_inp != model_dataspace_inp:
        return [ChangeDataSpace(dataspace_inp, model_dataspace_inp)]
    else:
        trans = []
        if dataspace_inp != model_dataspace_inp and dataspace_inp != -1 and model_dataspace_inp != -1:
            trans.append(ChangeDataSpace(dataspace_inp, model_dataspace_inp, include="inp"))
        elif dataspace_gt != model_dataspace_gt and dataspace_gt != -1 and model_dataspace_gt != -1:
            trans.append(ChangeDataSpace(dataspace_gt, model_dataspace_gt, include="gt"))
        return trans

