#pragma once

#ifdef __CUDACC__
#include "../utils/random.h"
#include "../utils/vec_math.h"
#endif

#ifndef __CUDACC__
#include "../headers/torch.h"
#endif
