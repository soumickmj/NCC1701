import os
import numpy as np
import ismrmrd
import ismrmrd.xsd

from utils.RAW.MeasurementsDS import MeasurementsDS 

class HeaderData():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class ISMRMRDReader(object):
    """description of class"""

    @staticmethod
    def readHeaderFromXML(headerXML):
        header = ismrmrd.xsd.CreateFromDocument(headerXML)
        enc = header.encoding[0]

        protocol_name = header.measurementInformation.protocolName
        trajectory = enc.trajectory

        # Matrix size
        eNx = enc.encodedSpace.matrixSize.x
        eNy = enc.encodedSpace.matrixSize.y
        eNz = enc.encodedSpace.matrixSize.z
        rNx = enc.reconSpace.matrixSize.x
        rNy = enc.reconSpace.matrixSize.y
        rNz = enc.reconSpace.matrixSize.z

        # Field of View
        eFOVx = enc.encodedSpace.fieldOfView_mm.x
        eFOVy = enc.encodedSpace.fieldOfView_mm.y
        eFOVz = enc.encodedSpace.fieldOfView_mm.z
        rFOVx = enc.reconSpace.fieldOfView_mm.x
        rFOVy = enc.reconSpace.fieldOfView_mm.y
        rFOVz = enc.reconSpace.fieldOfView_mm.z

        if enc.encodingLimits.kspace_encoding_step_0 != None:
            kspace_encoding_step_0_limit = enc.encodingLimits.kspace_encoding_step_0.maximum + 1
        else:
            kspace_encoding_step_0_limit = 1
        kspace_encoding_step_1_limit = enc.encodingLimits.kspace_encoding_step_1.maximum + 1 #data - y
        kspace_encoding_step_2_limit = enc.encodingLimits.kspace_encoding_step_2.maximum + 1 #slices

        if(kspace_encoding_step_2_limit != eNz):
            print('Slice Resolution: '+str((kspace_encoding_step_2_limit/eNz)*100))

        # Number of Slices, Reps, Contrasts, etc.
        ncoils = header.acquisitionSystemInformation.receiverChannels

        if enc.encodingLimits.slice != None:
            nslices = enc.encodingLimits.slice.maximum + 1
        else:
            nslices = 1

        if enc.encodingLimits.repetition != None:
            nreps = enc.encodingLimits.repetition.maximum + 1
        else:
            nreps = 1

        if enc.encodingLimits.average != None:
            naverage = enc.encodingLimits.average.maximum + 1
        else:
            naverage = 1

        if enc.encodingLimits.phase != None:
            nphase = enc.encodingLimits.phase.maximum + 1
        else:
            nphase = 1

        if enc.encodingLimits.segment != None:
            nsegment = enc.encodingLimits.segment.maximum + 1
        else:
            nsegment = 1

        if enc.encodingLimits.set_ != None:
            nsets = enc.encodingLimits.set_.maximum + 1
        else:
            nsets = 1

        if enc.encodingLimits.contrast != None:
            ncontrasts = enc.encodingLimits.contrast.maximum + 1
        else:
            ncontrasts = 1

        kspace_size_x = eNx*2
        kspace_size_y = kspace_encoding_step_1_limit
        kspace_size_z = kspace_encoding_step_2_limit

        measDS = MeasurementsDS(None, None, (kspace_size_x, kspace_size_y, kspace_size_z), (eNx, eNy, eNz), (rNx, rNy, rNz), (eFOVx, eFOVy, eFOVz), (rFOVx, rFOVy, rFOVz), ncoils, nslices, nreps, naverage, nphase, nsegment, nsets, ncontrasts, trajectory, protocol_name, header)
        return measDS

    @staticmethod
    def readFileHeader(filename):
        if not os.path.isfile(filename):
            print("%s is not a valid file" % filename)
            raise SystemExit
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        measDS = self.readHeaderFromXML(dset.read_xml_header())
        return measDS, dset

    @staticmethod
    def readData(filename):
        measDS, dset = self.readFileHeader(filename)

        print('Reading Measurement Data')

        measDS.noiseScans = np.zeros((measDS.nsegment, measDS.nsets, measDS.nreps, measDS.ncontrasts, measDS.nphase, measDS.naverage, measDS.nslices, measDS.ncoils, 1, 1, measDS.kspace_size_x), dtype=np.complex64)
        measDS.measurements =  np.zeros((measDS.nsegment, measDS.nsets, measDS.nreps, measDS.ncontrasts, measDS.nphase, measDS.naverage, measDS.nslices, measDS.ncoils, measDS.kspace_size_z, measDS.kspace_size_y, measDS.kspace_size_x), dtype=np.complex64)
        noiseScanAvl = False


        # loop through all the acquisitions 
        noiseScanData = np.array([])
        parallelCalibScans = np.array([])
        parallelCalibNImagingScans = np.array([])
        navigationData = np.array([])
        phaseCorrData = np.array([])
        hpFeedBackData = np.array([])
        dummyScanData = np.array([])
        rtFeedBackData = np.array([])
        measData = np.array(range(dset.number_of_acquisitions()))
        for acqnum in measData:
            acq = dset.read_acquisition(acqnum)
            avg = acq.idx.average
            phase = acq.idx.phase
            seg = acq.idx.segment
            set = acq.idx.set
            rep = acq.idx.repetition
            contrast = acq.idx.contrast
            slice = acq.idx.slice
            y = acq.idx.kspace_encode_step_1
            z = acq.idx.kspace_encode_step_2
    
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                noiseScanData = np.append(noiseScanData, [acqnum])
                noiseScanAvl = True
                measDS.noiseScans[seg, set, rep, contrast, phase, avg, slice, :, z, y, :] = acq.data

            elif acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                parallelCalibScans = np.append(parallelCalibScans, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING):
                parallelCalibNImagingScans = np.append(parallelCalibNImagingScans, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_NAVIGATION_DATA):
                navigationData = np.append(navigationData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_PHASECORR_DATA):
                phaseCorrData = np.append(phaseCorrData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_HPFEEDBACK_DATA):
                hpFeedBackData = np.append(hpFeedBackData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
                dummyScanData = np.append(dummyScanData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_RTFEEDBACK_DATA):
                rtFeedBackData = np.append(rtFeedBackData, [acqnum])
            else:
                # measurements[seg, set, rep, contrast, phase, avg, slice, :, z, y, 0:acq.data.shape[1]] = acq.data #Zeros after data
                #measurements[seg, set, rep, contrast, phase, avg, slice, :, z, y, -acq.data.shape[1]:] = acq.data #Zero before data
                measDS.measurements[seg, set, rep, contrast, phase, avg, slice, :, z, y, :] = acq.data #no zero filling

        if(not noiseScanAvl):
            measDS.noiseScans = None

        measData = np.setxor1d(measData,noiseScanData)
        measData = np.setxor1d(measData,parallelCalibScans)
        measData = np.setxor1d(measData,parallelCalibNImagingScans)
        measData = np.setxor1d(measData,navigationData)
        measData = np.setxor1d(measData,phaseCorrData)
        measData = np.setxor1d(measData,hpFeedBackData)
        measData = np.setxor1d(measData,dummyScanData)
        measData = np.setxor1d(measData,rtFeedBackData)

        return measDS

    @staticmethod
    def readNoise(filename):
        measDS, dset = self.readFileHeader(filename)

        print('Reading Noise Data')

        measDS.noiseScans = np.zeros((measDS.nsegment, measDS.nsets, measDS.nreps, measDS.ncontrasts, measDS.nphase, 2, measDS.nslices, measDS.ncoils, 1, measDS.eNx, int(measDS.eNx*4)), dtype=np.complex64) #TODO: Why 4 times eNx. Why avg 2 no idea
        measDS.measurements =  np.zeros((measDS.nsegment, 2, measDS.nreps, measDS.ncontrasts, measDS.nphase, measDS.naverage, measDS.nslices, measDS.ncoils, measDS.kspace_size_z, measDS.kspace_size_y, measDS.kspace_size_x//2), dtype=np.complex64)
        noiseScanAvl = False


        # loop through all the acquisitions 
        noiseScanData = np.array([])
        parallelCalibScans = np.array([])
        parallelCalibNImagingScans = np.array([])
        navigationData = np.array([])
        phaseCorrData = np.array([])
        hpFeedBackData = np.array([])
        dummyScanData = np.array([])
        rtFeedBackData = np.array([])
        measData = np.array(range(dset.number_of_acquisitions()))
        for acqnum in measData:
            acq = dset.read_acquisition(acqnum)
            avg = acq.idx.average
            phase = acq.idx.phase
            seg = acq.idx.segment
            set = acq.idx.set
            rep = acq.idx.repetition
            contrast = acq.idx.contrast
            slice = acq.idx.slice
            y = acq.idx.kspace_encode_step_1
            z = acq.idx.kspace_encode_step_2
    
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                noiseScanData = np.append(noiseScanData, [acqnum])
                noiseScanAvl = True
                measDS.noiseScans[seg, set, rep, contrast, phase, avg, slice, :, z, y, :] = acq.data

            elif acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                parallelCalibScans = np.append(parallelCalibScans, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING):
                parallelCalibNImagingScans = np.append(parallelCalibNImagingScans, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_NAVIGATION_DATA):
                navigationData = np.append(navigationData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_PHASECORR_DATA):
                phaseCorrData = np.append(phaseCorrData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_HPFEEDBACK_DATA):
                hpFeedBackData = np.append(hpFeedBackData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
                dummyScanData = np.append(dummyScanData, [acqnum])
            elif acq.isFlagSet(ismrmrd.ACQ_IS_RTFEEDBACK_DATA):
                rtFeedBackData = np.append(rtFeedBackData, [acqnum])
            else:
                 #measurements[seg, set, rep, contrast, phase, avg, slice, :, z, y, 0:acq.data.shape[1]] = acq.data #Zeros after data
                 measDS.measurements[seg, set, rep, contrast, phase, avg, slice, 0:acq.data.shape[0], z, y, :] = acq.data #Zeros after data
                #measurements[seg, set, rep, contrast, phase, avg, slice, :, z, y, -acq.data.shape[1]:] = acq.data #Zero before data
                #measurements[seg, set, rep, contrast, phase, avg, slice, :, z, y, :] = acq.data #no zero filling

        if(not noiseScanAvl):
            measDS.noiseScans = None

        measData = np.setxor1d(measData,noiseScanData)
        measData = np.setxor1d(measData,parallelCalibScans)
        measData = np.setxor1d(measData,parallelCalibNImagingScans)
        measData = np.setxor1d(measData,navigationData)
        measData = np.setxor1d(measData,phaseCorrData)
        measData = np.setxor1d(measData,hpFeedBackData)
        measData = np.setxor1d(measData,dummyScanData)
        measData = np.setxor1d(measData,rtFeedBackData)

        return measDS, measDS.noiseScans

    @staticmethod
    def readHeaderFromXMLExtended(headerXML):
        header = ismrmrd.xsd.CreateFromDocument(headerXML)
        headerData = HeaderData()

        sequenceParameters = header.sequenceParameters

        experimentalConditions = header.experimentalConditions
        headerData.H1resonanceFrequency_Hz = experimentalConditions.H1resonanceFrequency_Hz

        #Acquisition System info
        acquisitionSystemInformation = header.acquisitionSystemInformation
        headerData.systemVendor = acquisitionSystemInformation.systemVendor
        headerData.systemModel = acquisitionSystemInformation.systemModel
        headerData.systemFieldStrength_T = acquisitionSystemInformation.systemFieldStrength_T
        headerData.relativeReceiverNoiseBandwidth = acquisitionSystemInformation.relativeReceiverNoiseBandwidth
        headerData.receiverChannels = acquisitionSystemInformation.receiverChannels
        headerData.institutionName = acquisitionSystemInformation.institutionName
        headerData.stationName = acquisitionSystemInformation.stationName

        #Measurement Info
        measurementInformation = header.measurementInformation
        headerData.measurementID = measurementInformation.measurementID
        headerData.seriesDate = measurementInformation.seriesDate
        headerData.seriesTime = measurementInformation.seriesTime
        headerData.patientPosition = measurementInformation.patientPosition
        headerData.initialSeriesNumber= measurementInformation.initialSeriesNumber
        headerData.protocolName = measurementInformation.protocolName
        headerData.seriesDescription = measurementInformation.seriesDescription
        headerData.measurementDependency = []
        for md in measurementInformation.measurementDependency:
            dict = {'DependencyType':md.dependencyType, 'MeasurementID':md.measurementID}
            measurementDependency.append(dict)
        headerData.seriesInstanceUIDRoot = measurementInformation.seriesInstanceUIDRoot
        headerData.frameOfReferenceUID = measurementInformation.frameOfReferenceUID
        headerData.referencedImageSequenceInstID = []
        if(measurementInformation.referencedImageSequence is not None):
            for ris in measurementInformation.referencedImageSequence.referencedSOPInstanceUID:
                referencedImageSequenceInstID.append(ris)

        #Study Info
        studyInformation = header.studyInformation
        headerData.studyDate = studyInformation.studyDate
        headerData.studyTime = studyInformation.studyTime
        headerData.studyID = studyInformation.studyID
        headerData.accessionNumber = studyInformation.accessionNumber
        headerData.referringPhysicianName = studyInformation.referringPhysicianName
        headerData.studyDescription = studyInformation.studyDescription
        headerData.studyInstanceUID = studyInformation.studyInstanceUID

        #Subject Info
        subjectInformation = header.subjectInformation
        if(subjectInformation is not None):
            headerData.patientName = subjectInformation.patientName
            headerData.patientWeight_kg = subjectInformation.patientWeight_kg
            headerData.patientID = subjectInformation.patientID
            headerData.patientBirthdate = subjectInformation.patientBirthdate
            headerData.patientGender = subjectInformation.patientGender

        headerData.userParameters = header.userParameters
        headerData.version = header.version


        ##################
        #Encoding Info - Currently can only deal with one encoding
        #TODO: Handle multiple encoding
        enc = header.encoding[0]
        if(len(header.encoding) > 1):
            print('Multiple encoding detected. Not implimented yet')
            headerData.encodings = []
            for en in header.encoding:
                headerData.encodings.append(en)
            #But what to do with these encodings, not been taken care yet
            #May be to repeat all the things that are current done for 'enc'


        # Matrix size (Encoding Info)
        headerData.eNx = enc.encodedSpace.matrixSize.x
        headerData.eNy = enc.encodedSpace.matrixSize.y
        headerData.eNz = enc.encodedSpace.matrixSize.z
        headerData.rNx = enc.reconSpace.matrixSize.x
        headerData.rNy = enc.reconSpace.matrixSize.y
        headerData.rNz = enc.reconSpace.matrixSize.z

        # Field of View (mm) (Encoding Info)
        headerData.eFOVx = enc.encodedSpace.fieldOfView_mm.x
        headerData.eFOVy = enc.encodedSpace.fieldOfView_mm.y
        headerData.eFOVz = enc.encodedSpace.fieldOfView_mm.z
        headerData.rFOVx = enc.reconSpace.fieldOfView_mm.x
        headerData.rFOVy = enc.reconSpace.fieldOfView_mm.y
        headerData.rFOVz = enc.reconSpace.fieldOfView_mm.z

        # Other encoding info
        headerData.echoTrainLength = enc.echoTrainLength
        # Encoding Limits
        #kspace_encoding_step_0_min = enc.encodingLimits.kspace_encoding_step_0.minimum
        #kspace_encoding_step_0_max = enc.encodingLimits.kspace_encoding_step_0.maximum
        #kspace_encoding_step_0_center = enc.encodingLimits.kspace_encoding_step_0.center
        headerData.kspace_encoding_step_1_min = enc.encodingLimits.kspace_encoding_step_1.minimum
        headerData.kspace_encoding_step_1_max = enc.encodingLimits.kspace_encoding_step_1.maximum
        headerData.kspace_encoding_step_1_center = enc.encodingLimits.kspace_encoding_step_1.center
        headerData.kspace_encoding_step_2_min = enc.encodingLimits.kspace_encoding_step_2.minimum
        headerData.kspace_encoding_step_2_max = enc.encodingLimits.kspace_encoding_step_2.maximum
        headerData.kspace_encoding_step_2_center = enc.encodingLimits.kspace_encoding_step_2.center
        headerData.average_min = enc.encodingLimits.average.minimum
        headerData.average_max = enc.encodingLimits.average.maximum
        headerData.average_center = enc.encodingLimits.average.center
        headerData.slice_min = enc.encodingLimits.slice.minimum
        headerData.slice_max = enc.encodingLimits.slice.maximum
        headerData.slice_center = enc.encodingLimits.slice.center
        headerData.contrast_min = enc.encodingLimits.contrast.minimum
        headerData.contrast_max = enc.encodingLimits.contrast.maximum
        headerData.contrast_center = enc.encodingLimits.contrast.center
        headerData.phase_min = enc.encodingLimits.phase.minimum
        headerData.phase_max = enc.encodingLimits.phase.maximum
        headerData.phase_center = enc.encodingLimits.phase.center
        headerData.repetition_min = enc.encodingLimits.repetition.minimum
        headerData.repetition_max = enc.encodingLimits.repetition.maximum
        headerData.repetition_center = enc.encodingLimits.repetition.center
        #set_min = enc.encodingLimits.set.minimum
        #set_max = enc.encodingLimits.set.maximum
        #set_center = enc.encodingLimits.set.center
        headerData.segment_min = enc.encodingLimits.segment.minimum
        headerData.segment_max = enc.encodingLimits.segment.maximum
        headerData.segment_center = enc.encodingLimits.segment.center

        # Number of Slices, Reps, Contrasts, etc. (Encoding Info)
        try:
            headerData.ncoils = header.acquisitionSystemInformation.receiverChannels
        except:
            headerData.ncoils = 1

        if enc.encodingLimits.slice != None:
            headerData.nslices = enc.encodingLimits.slice.maximum + 1
        else:
            headerData.nslices = 1

        if enc.encodingLimits.repetition != None:
            headerData.nreps = enc.encodingLimits.repetition.maximum + 1
        else:
            headerData.nreps = 1

        if enc.encodingLimits.contrast != None:
            headerData.ncontrasts = enc.encodingLimits.contrast.maximum + 1
        else:
            headerData.ncontrasts = 1

        # trajectoryType Info (Encoding Info)
        trajectoryType = enc.trajectory
        if(enc.trajectoryDescription is not None):
            headerData.trajectoryIdentifier = enc.trajectoryDescription.identifier
            headerData.trajectoryUserParameterLong = []
            for upL in enc.trajectoryDescription.userParameterLong:
                dict = {upL.name : upL.value}
                headerData.trajectoryUserParameterLong.append(dict)
            headerData.trajectoryUserParameterDouble = []
            for upD in enc.trajectoryDescription.userParameterDouble:
                dict = {upD.name : upD.value}
                headerData.trajectoryUserParameterDouble.append(dict)
            headerData.sequenceParametersTE = []
            for te in header.sequenceParameters.TE:
                headerData.sequenceParametersTE.append(te)
            headerData.trajectoryDescription.comment = enc.trajectoryDescription.comment

        # Parallel Imaging Info (Encoding Info)
        parallelImagingAccelerationFactor = enc.parallelImaging.accelerationFactor
        headerData.accelerationFactorType_kspace_encoding_step_1 = parallelImagingAccelerationFactor.kspace_encoding_step_1
        headerData.accelerationFactorType_kspace_encoding_step_2 = parallelImagingAccelerationFactor.kspace_encoding_step_2
        headerData.parallelImagingCalibrationModeType = enc.parallelImaging.calibrationMode
        headerData.parallelImagingInterleavingDimensionType = enc.parallelImaging.interleavingDimension

        ##########################################
        # Sequence Parameters Info
        headerData.sequenceParametersTR = []
        for tr in header.sequenceParameters.TR:
            headerData.sequenceParametersTR.append(tr)
        headerData.sequenceParametersTE = []
        for te in header.sequenceParameters.TE:
            headerData.sequenceParametersTE.append(te)
        headerData.sequenceParametersTI = []
        for ti in header.sequenceParameters.TI:
            headerData.sequenceParametersTI.append(ti)
        headerData.sequenceParametersflipAngle_deg = [0]
        for ang in header.sequenceParameters.flipAngle_deg:
            headerData.sequenceParametersflipAngle_deg.append(ang)


        # User Parameters Info
        headerData.userParameterLong = [] #Contains Embedded Ref Lines E1 - TODO May need to handle this for GRAPPA
        for upL in header.userParameters.userParameterLong:
            dict = {upL.name : upL.value}
            headerData.userParameterLong.append(dict)
        headerData.userParameterDouble = [] #Contains MaxwellCoofficients for each 16 coil - in chomp liver vibe
        for upD in header.userParameters.userParameterDouble:
            dict = {upD.name : upD.value}
            headerData.userParameterDouble.append(dict)
        headerData.userParameterString = []
        for upS in header.userParameters.userParameterString:
            dict = {upS.name : upS.value}
            headerData.userParameterString.append(dict)
        headerData.userParameterBase64 = []
        for upB in header.userParameters.userParameterBase64:
            dict = {upB.name : upB.value}
            headerData.userParameterBase64.append(dict)

        headerDict = vars(headerData)

        return headerDict