#ifndef _TRILINEARTEST__H_
#define _TRILINEARTEST__H_

//--------------------------------------------------------------------------

#include "Shared.h"

//--------------------------------------------------------------------------

TEST(Texture3DCpu, FloatTrilinearFetch0)
{
    // initialise float 3d texture
    Texture3DCpu<float> texture;
    texture.SetData(float_data_res, &float_data_0[0]);

    float val0 = texture.Eval(0.0f, 0.0f, 0.0f);
    float val1 = texture.Eval(1.0f, 1.0f, 1.0f);
    float val2 = texture.Eval(0.5f, 0.5f, 0.5f);


    // expected results
    float expected_v0 = 0.0f;
    float expected_v1 = 0.0f;
    float expected_v2 = 0.0f;


    EXPECT_EQ(val0, expected_v0);
    EXPECT_EQ(val1, expected_v1);
    EXPECT_EQ(val2, expected_v2);
}

//--------------------------------------------------------------------------

TEST(Texture3DCpu, FloatTrilinearFetch1)
{
    // initialise float 3d texture
    Texture3DCpu<float> texture;
    texture.SetData(float_data_res, &float_data_1[0]);

    float val0 = texture.Eval(0.0f, 0.0f, 0.0f);
    float val1 = texture.Eval(1.0f, 1.0f, 1.0f);
    float val2 = texture.Eval(0.5f, 0.5f, 0.5f);


    // expected results
    float expected_v0 = 1.0f;
    float expected_v1 = 1.0f;
    float expected_v2 = 1.0f;


    EXPECT_EQ(val0, expected_v0);
    EXPECT_EQ(val1, expected_v1);
    EXPECT_EQ(val2, expected_v2);
}


//--------------------------------------------------------------------------

TEST(Texture3DCpu, FloatTrilinearFetch2)
{
    // initialise float 3d texture
    Texture3DCpu<float> texture;
    texture.SetData(float_data_res, &float_data_2[0]);

    float val0 = texture.Eval(0.0f, 0.0f, 0.0f);
    float val1 = texture.Eval(1.0f, 1.0f, 1.0f);
    float val2 = texture.Eval(0.5f, 0.5f, 0.5f*((1.0f/float_data_res)+(2.0f/float_data_res)));


    // expected results
    float expected_v0 = 1.0f;
    float expected_v1 = 10.0f;
    float expected_v2 = 2.5f;


    EXPECT_EQ(val0, expected_v0);
    EXPECT_EQ(val1, expected_v1);
    EXPECT_EQ(val2, expected_v2);
}


//--------------------------------------------------------------------------

#endif // _TRILINEARTEST__H_
