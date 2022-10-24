//#include "typedefs.h"
#include "common_utils/tools.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "LFDataUtils/LFDataUtilParams.h"
#include "LFReconstruct/LFReconstructParams.h"

using namespace std;
namespace po = boost::program_options;

/*! Function to parse a settings file and return a variosu settings objects required by different modules
  \param filename Full path to configuration file
  \param settings Destination Settings variable
  \param reconSettings Destination Settings variable for LF reconstruction
  \param h Flag to show all options as a help menu instead of parsing data
*/
void parse_settings(string filename, LFDataUtilSettings &settings, LFReconstructSettings &reconSettings, bool h);

/*! Function to parse a settings file.
  \param filename Full path to configuration file
  \param settings Destination Settings variable
  \param h Flag to show all options as a help menu instead of parsing data
*/
void parse_settings(string filename, LFDataUtilSettings &settings, bool h);

/*! Function to parse a Depth reconstruction settings and return a depthe_settings variable
  \param filename Full path to configuration file
  \param settings Destination Settings variable
  \param help Flag to show all options as a help menu instead of parsing data
*/
void parseLFReconstructParams(LFReconstructSettings &settings,  po::variables_map vm);

void parseLFDataUtilParams(LFDataUtilSettings &settings,  po::variables_map vm);

/*! Function to parse a calibration settings file and return a calibration_settings variable
  which can be directly passed to an saRefocus constructor.
  \param filename Full path to configuration file
  \param settings Destination calibration_settings variable
  \param help Flag to show all options as a help menu instead of parsing data
*/
//void parse_calibration_settings(string filename, calibration_settings &settings, bool help);
