#pragma once

#include <cstring>
#include <sstream> 
#include <iomanip>



    /**
     * \brief generate a string designed for filenames that includes a number coded on a specified digit number. To sumarize, this fuction builds : "filename" + number (on x digit) + "extension". For example : "output"+24(5)+".jpg" => "output00024.jpg". 
     * \param filename: prefix of the resulting string.
     * \param number : number to be read on the resulting string.
     * \param digitCard : number of digit to code number.
     * \param extension : optional extension (the user should specify the caracter '.')
     * \return  : "filename" + number (digit) + "extension"
     */
    inline std::string stringEnumerator(const std::string &filename, const unsigned int number, const unsigned int digitCard, const std::string extension = ""){
      std::ostringstream oss;
      oss << filename;
      oss << std::setfill('0') << std::setw(digitCard) << number ;
      oss << extension;

      return oss.str();
    }
