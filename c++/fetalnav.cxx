#include <iostream>
#include <cstring>
#include <vector>

#include <itkImageFileReader.h>
#include <itkImportImageFilter.h>
#include <itkImageFileWriter.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

const unsigned int ImageDimension = 2;
typedef unsigned char GrayPixelType;
typedef itk::Image< GrayPixelType, ImageDimension > GrayImageType;
typedef GrayImageType::SizeType SizeType;
typedef itk::ImageFileReader< GrayImageType >  ReaderType;
typedef itk::ImportImageFilter< GrayPixelType, 2 > ImportFilterType2;
typedef itk::ImportImageFilter< GrayPixelType, 3 > ImportFilterType3;
typedef itk::ImageFileWriter< ImportFilterType2::OutputImageType >  WriterType2;
typedef itk::ImageFileWriter< ImportFilterType3::OutputImageType >  WriterType3;

namespace py = pybind11;

int main(int argc, char* argv[])
{
  
  /// Check inputs for validity
  if (argc < 2) 
  {
    std::cout << "Usage:" << std::endl;
    std::cout << argv[0] <<"\t<fetal ultrasound image (in polar coordinates)>" << std::endl;
    return 0;
  }

  /// Read image as a 2D 
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );
  reader->Update();
  GrayImageType::Pointer image_itk = reader->GetOutput();
  /// Write the input for comodity
  WriterType2::Pointer w0 = WriterType2::New();
  w0->SetFileName("/tmp/input.mha");
  w0->SetInput(image_itk);
  w0->Update();

  /// Create interpreter
  py::scoped_interpreter guard{};
  /// Import module and inference function
  py::object module = py::module::import("fetalnav");    
  py::object getprediction = module.attr("getprediction");
  /// Create a numpy array containing the image scalars
  /// Input dimensions are swapped as ITK and numpy have inverted orders
  std::vector <unsigned long> dims = {image_itk->GetLargestPossibleRegion().GetSize()[1], 
                            image_itk->GetLargestPossibleRegion().GetSize()[0]};
  py::array numpyarray(dims, static_cast<GrayPixelType*>(image_itk->GetBufferPointer()));
  /// Call network using the numpy array
  py::tuple predictions = py::tuple(getprediction(numpyarray));
  /// Extract network output confidences (list of floating points)
  py::tuple confidences = py::tuple(predictions[0]);
  /// Extract network output responses (ND numpy array)
  py::array responses   = py::array(predictions[1]);
  /// Print the network output 
  py::print(py::str(confidences));

  /// Define the image import parameters
  ImportFilterType2::SizeType imagesize;
  imagesize[0] = responses.shape(2);
  imagesize[1] = responses.shape(1);
  ImportFilterType2::IndexType start;
  start.Fill(0);
  ImportFilterType2::RegionType region;
  region.SetIndex(start);
  region.SetSize(imagesize);

  for (unsigned int idx=0; idx<responses.shape(0); idx++)
  {
    /// Define import filter
    ImportFilterType2::Pointer importer = ImportFilterType2::New();
    importer->SetOrigin( image_itk->GetOrigin() );
    importer->SetSpacing( image_itk->GetSpacing() );
    importer->SetDirection( image_itk->GetDirection() );
    importer->SetRegion(region);
    /// Separate the regional scalar buffer
    /// @todo check if a memcpy is necessary here  
    GrayPixelType* localbuffer = static_cast<GrayPixelType*>(responses.mutable_data(idx));
    /// Import the buffer
    importer->SetImportPointer(localbuffer, imagesize[0] * imagesize[1], false);
    importer->Update();
    /// Disconnect the output from the filter
    /// @todo Check if that is sufficient to release the numpy buffer, or if the buffer needs to obe memcpy'ed 
    GrayImageType::Pointer responsemap = importer->GetOutput();
    responsemap->DisconnectPipeline();
    /// Re-assign the metadata to the output image
    responsemap->SetMetaDataDictionary(image_itk->GetMetaDataDictionary());
    /// Write the selected response map
    WriterType2::Pointer w = WriterType2::New();
    std::ostringstream fname; fname << "/tmp/responses_" << idx << ".mha";
    w->SetFileName(fname.str());
    w->SetInput(responsemap);
    w->Update();
  }

  /// Define the image import parameters
  ImportFilterType3::SizeType imagesize3;
  imagesize3[0] = responses.shape(2);
  imagesize3[1] = responses.shape(1);
  imagesize3[2] = responses.shape(0);
  ImportFilterType3::IndexType start3;
  start3.Fill(0);
  ImportFilterType3::RegionType region3;
  region3.SetIndex(start3);
  region3.SetSize(imagesize3);
  /// Define import filter
  ImportFilterType3::Pointer importer = ImportFilterType3::New();
  double origin3[3] = {image_itk->GetOrigin()[0], image_itk->GetOrigin()[1], 0};
  importer->SetOrigin(origin3);
  double spacing3[3] = {image_itk->GetSpacing()[0], image_itk->GetSpacing()[1], 1};
  importer->SetSpacing(spacing3);
  importer->SetRegion(region3);
  /// Select the entire scalar buffer
  /// @todo check if a memcpy is necessary here  
  GrayPixelType* localbuffer = static_cast<GrayPixelType*>(responses.mutable_data());
  /// Import the buffer
  importer->SetImportPointer(localbuffer, imagesize3[0] * imagesize3[1] * imagesize3[2], false);
  importer->Update();
  /// Disconnect the output from the filter
  /// @todo Check if that is sufficient to release the numpy buffer, or if the buffer needs to obe memcpy'ed 
  ImportFilterType3::OutputImageType::Pointer responsemap = importer->GetOutput();
  responsemap->DisconnectPipeline();
  /// Re-assign the metadata to the output image
  responsemap->SetMetaDataDictionary(image_itk->GetMetaDataDictionary());
  /// Write the entire response map
  WriterType3::Pointer w = WriterType3::New();
  w->SetFileName("/tmp/responses.mha");
  w->SetInput(responsemap);
  w->Update();

  return EXIT_SUCCESS;  
}
