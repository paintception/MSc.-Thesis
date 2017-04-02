import dicom
import os
import numpy

from matplotlib import pyplot as plt

PathDicom = "/home/matthia/Downloads/DOI/"

lstFilesDCM = []

for dirName, subdirList, fileList in os.walk(PathDicom):
	for filename in fileList:
		if ".dcm" in filename.lower():  # check whether the file's DICOM
			lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dicom.read_file(lstFilesDCM[0])# Load dimensions based on the number of rows, columns, and slices (along the Z axis)

print type(RefDs)

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
	# read the file
	ds = dicom.read_file(filenameDCM)
	# store the raw image data
	ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel

plt.figure(dpi=300)
plt.axes().set_aspect('equal', 'datalim')
plt.set_cmap(plt.gray())
plt.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 80]))