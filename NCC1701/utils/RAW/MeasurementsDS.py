class MeasurementsDS(object):
    """description of class"""

    def __init__(self, measData, noiseScans, kSpaceSize, encodedSpace, reconSpace, encodedFOV, reconFOV, ncoils, nslices, nreps, naverage, nphase, nsegment, nsets, ncontrasts, trajectory, protocol_name, header=None):
        #header only applicable for ISMRMRD format

        super().__init__()

        self.measData = measData
        self.noiseScans = noiseScans

        self.kspace_size_x = kSpaceSize[0]
        self.kspace_size_y = kSpaceSize[1]
        self.kspace_size_z = kSpaceSize[2]

        self.eNx = encodedSpace[0]
        self.eNy = encodedSpace[1]
        self.eNz = encodedSpace[2]
        self.rNx = reconSpace[0]
        self.rNy = reconSpace[1]
        self.rNz = reconSpace[2]

        self.eFOVx = encodedFOV[0]
        self.eFOVy = encodedFOV[1]
        self.eFOVz = encodedFOV[2]
        self.rFOVx = reconFOV[0]
        self.rFOVy = reconFOV[1]
        self.rFOVz = reconFOV[2]

        self.ncoils = ncoils
        self.nslices = nslices
        self.nreps = nreps
        self.naverage = naverage
        self.nphase = nphase
        self.nsegment = nsegment
        self.nsets = nsets
        self.ncontrasts = ncontrasts

        self.trajectory = trajectory
        self.protocol_name = protocol_name
        self.header = header




