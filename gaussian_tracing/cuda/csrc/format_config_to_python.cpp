#include "flags.h"
#include <fstream>
#include <iostream>

int main() {
    std::cout << "CONFIG = " << CONFIG << "\n";
    std::cout << "ALPHA_THRESHOLD = " << ALPHA_THRESHOLD << "\n";
    std::cout << "T_THRESHOLD = " << T_THRESHOLD << "\n";
    std::cout << "MAX_BOUNCES = " << MAX_BOUNCES << "\n";
    #if REMAINING_COLOR_ESTIMATION == ASSUME_SAME_AS_FOREGROUND
        std::cout << "REMAINING_COLOR_ESTIMATION = 'ASSUME_SAME_AS_FOREGROUND'" << "\n";
    #elif REMAINING_COLOR_ESTIMATION == IGNORE_OCCLUSION
        std::cout << "REMAINING_COLOR_ESTIMATION = 'IGNORE_OCCLUSION'" << "\n";
    #elif REMAINING_COLOR_ESTIMATION == STOCHASTIC_SAMPLING
        std::cout << "REMAINING_COLOR_ESTIMATION = 'STOCHASTIC_SAMPLING'" << "\n";
    #elif REMAINING_COLOR_ESTIMATION == RENDER_BUT_DETACH
        std::cout << "REMAINING_COLOR_ESTIMATION = 'RENDER_BUT_DETACH'" << "\n";
    #elif REMAINING_COLOR_ESTIMATION == NO_ESTIMATION
        std::cout << "REMAINING_COLOR_ESTIMATION = 'NO_ESTIMATION'" << "\n";
    #endif
    #if STORAGE_MODE == NO_STORAGE
        std::cout << "STORAGE_MODE = 'NO_STORAGE'" << "\n";
    #elif STORAGE_MODE == PER_PIXEL_LINKED_LIST
        std::cout << "STORAGE_MODE = 'PER_PIXEL_LINKED_LIST'" << "\n";
    #endif
    std::cout << "MAX_ITERATIONS = " << MAX_ITERATIONS << "\n";                
    #if ALPHA_RESCALE == true
        std::cout << "ALPHA_RESCALE = " << "True" << "\n";
    #else
        std::cout << "ALPHA_RESCALE = " << "False" << "\n";
    #endif
    std::cout << "EXP_POWER = " << EXP_POWER << "\n";
    #if T_GRADS == true
        std::cout << "T_GRADS = " << "True" << "\n";
    #else
        std::cout << "T_GRADS = " << "False" << "\n";
    #endif
    #if ATTACH_NORMALS == true
        std::cout << "ATTACH_NORMALS = " << "True" << "\n";
    #else 
        std::cout << "ATTACH_NORMALS = " << "False" << "\n";
    #endif
    #if ATTACH_POSITION == true
        std::cout << "ATTACH_POSITION = " << "True" << "\n";
    #else 
        std::cout << "ATTACH_POSITION = " << "False" << "\n";
    #endif
    #if ATTACH_F0 == true
        std::cout << "ATTACH_F0 = " << "True" << "\n";
    #else
        std::cout << "ATTACH_F0 = " << "False" << "\n";
    #endif
    #if ATTACH_ROUGHNESS == true
        std::cout << "ATTACH_ROUGHNESS = " << "True" << "\n";
    #else
        std::cout << "ATTACH_ROUGHNESS = " << "False" << "\n";
    #endif
    #if ATTACH_SPECULAR == true
        std::cout << "ATTACH_SPECULAR = " << "True" << "\n";
    #else
        std::cout << "ATTACH_SPECULAR = " << "False" << "\n";
    #endif
    #if ATTACH_ALBEDO == true
        std::cout << "ATTACH_ALBEDO = " << "True" << "\n";
    #else
        std::cout << "ATTACH_ALBEDO = " << "False" << "\n";
    #endif
    #if ATTACH_METALNESS == true
        std::cout << "ATTACH_METALNESS = " << "True" << "\n";
    #else
        std::cout << "ATTACH_METALNESS = " << "False" << "\n";
    #endif
    #if POSITION_FROM_EXPECTED_TERMINATION_DEPTH == true
        std::cout << "POSITION_FROM_EXPECTED_TERMINATION_DEPTH = " << "True" << "\n";
    #else
        std::cout << "POSITION_FROM_EXPECTED_TERMINATION_DEPTH = " << "False" << "\n";
    #endif
    #if ENABLE_DEBUG_DUMP == true
        std::cout << "ENABLE_DEBUG_DUMP = " << ENABLE_DEBUG_DUMP << "\n";
    #else
        std::cout << "ENABLE_DEBUG_DUMP = " << "None" << "\n";
    #endif
    std::cout << "DEBUG_DUMP_PIXEL_ID = " << DEBUG_DUMP_PIXEL_ID << "\n";
    #if DEBUG_SECOND_BOUNCE == true
        std::cout << "DEBUG_SECOND_BOUNCE = " << "True" << "\n";
    #else
        std::cout << "DEBUG_SECOND_BOUNCE = " << "False" << "\n";
    #endif
    #if GLOBAL_SORT == true
        std::cout << "GLOBAL_SORT = " << "True" << "\n";
    #else
        std::cout << "GLOBAL_SORT = " << "False" << "\n";
    #endif
    #if ACTIVATION_IN_CUDA == true 
        std::cout << "ACTIVATION_IN_CUDA = " << "True" << "\n";
    #else
        std::cout << "ACTIVATION_IN_CUDA = " << "False" << "\n";
    #endif
    #if OPACITY_MODULATION == true
        std::cout << "OPACITY_MODULATION = " << "True" << "\n";
    #else
        std::cout << "OPACITY_MODULATION = " << "False" << "\n";
    #endif
    #if USE_EPANECHNIKOV_KERNEL == true
        std::cout << "USE_EPANECHNIKOV_KERNEL = " << "True" << "\n";
    #else
        std::cout << "USE_EPANECHNIKOV_KERNEL = " << "False" << "\n";
    #endif
    #if ALLOW_OPACITY_ABOVE_1 == true
        std::cout << "ALLOW_OPACITY_ABOVE_1 = " << "True" << "\n";
    #else
        std::cout << "ALLOW_OPACITY_ABOVE_1 = " << "False" << "\n";
    #endif
    #if SQUARE_KERNEL == true
        std::cout << "SQUARE_KERNEL = " << "True" << "\n";
    #else
        std::cout << "SQUARE_KERNEL = " << "False" << "\n";
    #endif
    #if OPTIMIZE_EXP_POWER == true
        std::cout << "OPTIMIZE_EXP_POWER = " << "True" << "\n";
    #else
        std::cout << "OPTIMIZE_EXP_POWER = " << "False" << "\n";
    #endif
    #if USE_LEVEL_OF_DETAIL == true
        std::cout << "USE_LEVEL_OF_DETAIL = " << "True" << "\n";
    #else
        std::cout << "USE_LEVEL_OF_DETAIL = " << "False" << "\n";
    #endif
    #if USE_GT_BRDF == true
        std::cout << "USE_GT_BRDF = " << "True" << "\n";
    #else
        std::cout << "USE_GT_BRDF = " << "False" << "\n";
    #endif
    #if TONEMAP == true
        std::cout << "TONEMAP = " << "True" << "\n";
    #else
        std::cout << "TONEMAP = " << "False" << "\n";
    #endif
    #if REFLECTION_RAY_FROM_GT_NORMAL == true
        std::cout << "REFLECTION_RAY_FROM_GT_NORMAL = " << "True" << "\n";
    #else
        std::cout << "REFLECTION_RAY_FROM_GT_NORMAL = " << "False" << "\n";
    #endif
    #if REFLECTION_RAY_FROM_GT_POSITION == true
        std::cout << "REFLECTION_RAY_FROM_GT_POSITION = " << "True" << "\n";
    #else
        std::cout << "REFLECTION_RAY_FROM_GT_POSITION = " << "False" << "\n";
    #endif
    #if SURFACE_BRDF_FROM_GT_F0 == true
        std::cout << "SURFACE_BRDF_FROM_GT_F0 = " << "True" << "\n";
    #else
        std::cout << "SURFACE_BRDF_FROM_GT_F0 = " << "False" << "\n";
    #endif
    #if SURFACE_BRDF_FROM_GT_ROUGHNESS == true
        std::cout << "SURFACE_BRDF_FROM_GT_ROUGHNESS = " << "True" << "\n";
    #else
        std::cout << "SURFACE_BRDF_FROM_GT_ROUGHNESS = " << "False" << "\n";
    #endif
    #if REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE == true
        std::cout << "REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE = " << "True" << "\n";
    #else
        std::cout << "REFLECTIONS_FROM_GT_GLOSSY_IRRADIANCE = " << "False" << "\n";
    #endif
    #if SAVE_LUT_IMAGES == true
        std::cout << "SAVE_LUT_IMAGES = " << "True" << "\n";
    #else
        std::cout << "SAVE_LUT_IMAGES = " << "False" << "\n";
    #endif
    #if SAVE_RAY_IMAGES == true
        std::cout << "SAVE_RAY_IMAGES = " << "True" << "\n";
    #else
        std::cout << "SAVE_RAY_IMAGES = " << "False" << "\n";
    #endif
    std::cout << "MAX_ALPHA = " << MAX_ALPHA << "\n";
    #if ATTACH_POSITION == true
        std::cout << "ATTACH_POSITION = " << "True" << "\n";
    #else 
        std::cout << "ATTACH_POSITION = " << "False" << "\n";
    #endif
    #if ATTACH_NORMALS == true
        std::cout << "ATTACH_NORMALS = " << "True" << "\n";
    #else 
        std::cout << "ATTACH_NORMALS = " << "False" << "\n";
    #endif
    #if ATTACH_F0 == true
        std::cout << "ATTACH_F0 = " << "True" << "\n";
    #else
        std::cout << "ATTACH_F0 = " << "False" << "\n";
    #endif
    #if ATTACH_ROUGHNESS == true
        std::cout << "ATTACH_ROUGHNESS = " << "True" << "\n";
    #else
        std::cout << "ATTACH_ROUGHNESS = " << "False" << "\n";
    #endif
    #if ATTACH_SPECULAR == true
        std::cout << "ATTACH_SPECULAR = " << "True" << "\n";
    #else
        std::cout << "ATTACH_SPECULAR = " << "False" << "\n";
    #endif
    #if ATTACH_ALBEDO == true
        std::cout << "ATTACH_ALBEDO = " << "True" << "\n";
    #else
        std::cout << "ATTACH_ALBEDO = " << "False" << "\n";
    #endif
    #if ATTACH_METALNESS == true
        std::cout << "ATTACH_METALNESS = " << "True" << "\n";
    #else
        std::cout << "ATTACH_METALNESS = " << "False" << "\n";
    #endif
    #if SAVE_ALL_MAPS == true
        std::cout << "SAVE_ALL_MAPS = " << "True" << "\n";
    #else
        std::cout << "SAVE_ALL_MAPS = " << "False" << "\n";
    #endif

    #if NORMALIZE_NORMAL_MAP == true
        std::cout << "NORMALIZE_NORMAL_MAP = " << "True" << "\n";
    #else
        std::cout << "NORMALIZE_NORMAL_MAP = " << "False" << "\n";
    #endif
    #if PROJECT_POSITION_TO_RAY == true
        std::cout << "PROJECT_POSITION_TO_RAY = " << "True" << "\n";
    #else
        std::cout << "PROJECT_POSITION_TO_RAY = " << "False" << "\n";
    #endif
    std::cout << "USE_RUSSIAN_ROULETTE = " << USE_RUSSIAN_ROULETTE << "\n";
    std::cout << "REFLECTION_VALID_NORMAL_MIN_NORM = " << REFLECTION_VALID_NORMAL_MIN_NORM << "\n";

    std::cout << "INIT_F0 = " << INIT_F0 << "\n";
    std::cout << "INIT_ROUGHNESS = " << INIT_ROUGHNESS << "\n";

    #if RGB_UNWEIGHTED_LOSS == true
        std::cout << "RGB_UNWEIGHTED_LOSS = " << "True" << "\n";
    #else
        std::cout << "RGB_UNWEIGHTED_LOSS = " << "False" << "\n";
    #endif
    #if SKIP_BACKFACING_NORMALS == true
        std::cout << "SKIP_BACKFACING_NORMALS = " << "True" << "\n";
    #else
        std::cout << "SKIP_BACKFACING_NORMALS = " << "False" << "\n";
    #endif
    std::cout << "SKIP_BACKFACING_REFLECTION_VALID_NORMAL_MIN_NORM = " << SKIP_BACKFACING_REFLECTION_VALID_NORMAL_MIN_NORM << "\n";

    std::cout << "LOD_KERNEL_EXPONENT = " << LOD_KERNEL_EXPONENT << "\n";

    std::cout << "SAVE_LOD_IMAGES = " << SAVE_LOD_IMAGES << "\n";
    std::cout << "SAVE_HIT_STATS = " << SAVE_HIT_STATS << "\n";

    std::cout << "SURFACE_EPS = " << SURFACE_EPS << "\n";

    std::cout << "USE_GRADIENT_SCALING = " << USE_GRADIENT_SCALING << "\n";

    std::cout << "RELU_INSTEAD_OF_SOFTPLUS = " << RELU_INSTEAD_OF_SOFTPLUS << "\n";
    std::cout << "CLIPPED_RELU_INSTEAD_OF_SIGMOID = " << CLIPPED_RELU_INSTEAD_OF_SIGMOID << "\n";
}   
