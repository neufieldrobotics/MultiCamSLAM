/******************************************************************************
 * Author: Laurent Kneip                                                      *
 * Contact: kneip.laurent@gmail.com                                           *
 * License: Copyright (c) 2013 Laurent Kneip, ANU. All rights reserved.       *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 * * Redistributions of source code must retain the above copyright           *
 * notice, this list of conditions and the following disclaimer.              *
 * * Redistributions in binary form must reproduce the above copyright        *
 * notice, this list of conditions and the following disclaimer in the        *
 * documentation and/or other materials provided with the distribution.       *
 * * Neither the name of ANU nor the names of its contributors may be         *
 * used to endorse or promote products derived from this software without     *
 * specific prior written permission.                                         *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"*
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE  *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE *
 * ARE DISCLAIMED. IN NO EVENT SHALL ANU OR THE CONTRIBUTORS BE LIABLE        *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL *
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT         *
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY  *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF     *
 * SUCH DAMAGE.                                                               *
 ******************************************************************************/

#include "MCSlam/time_measurement.hpp"

#ifdef WIN32

#include <sys\timeb.h>

void
gettimeofday( timeval * timeofday, int dummy)
{
  struct timeb time;
  ftime(&time);
  timeofday->tv_sec = (int) time.time;
  timeofday->tv_usec = 1000 * (int) time.millitm;
}

#endif

timeval
timeval_minus( const struct timeval &t1, const struct timeval &t2 )
{
  timeval ret;
  ret.tv_sec = t1.tv_sec - t2.tv_sec;
  if( t1.tv_usec < t2.tv_usec )
  {
    ret.tv_sec--;
    ret.tv_usec = t1.tv_usec - t2.tv_usec + 1000000;
  }
  else
    ret.tv_usec = t1.tv_usec - t2.tv_usec;

  return ret;
}

