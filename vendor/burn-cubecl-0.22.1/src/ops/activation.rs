use crate::CubeBackend;
use crate::kernel::{FloatUnaryOp, FloatUnaryOpFamily, launch_unary_float};
use burn_backend::{
    Scalar, cubecl::dtype_to_storage_type, ops::ActivationOps, tensor::FloatTensor,
};
use cubecl::prelude::*;

use crate::kernel::unary_basic::{self, BasicFloatUnaryKind};

struct LeakyRelu;

#[cube]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for LeakyRelu {
    type Options = InputScalar;

    fn execute(input: Vector<F, N>, slope: &Self::Options) -> Vector<F, N> {
        let negative = input.less_than(&Vector::new(F::new(0.0_f32)));
        select_many(negative, input * Vector::new(slope.get::<F>()), input)
    }
}

impl FloatUnaryOpFamily for LeakyRelu {
    type Options = InputScalar;
    type Unary<F: Float, N: Size> = Self;
}

struct Gelu;

#[cube]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for Gelu {
    type Options = InputScalar;

    fn execute(input: Vector<F, N>, sqrt_two: &Self::Options) -> Vector<F, N> {
        let cdf =
            Vector::erf(input / Vector::new(sqrt_two.get::<F>())) + Vector::new(F::new(1.0_f32));
        (input * cdf) / Vector::new(F::new(2.0_f32))
    }
}

impl FloatUnaryOpFamily for Gelu {
    type Options = InputScalar;
    type Unary<F: Float, N: Size> = Self;
}

impl ActivationOps<Self> for CubeBackend {
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch(tensor, |_| BasicFloatUnaryKind::Relu)
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let dtype = dtype_to_storage_type(tensor.dtype);
        launch_unary_float::<Gelu, _>(tensor, |_| {
            InputScalar::new(core::f64::consts::SQRT_2, dtype)
        })
    }

    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        let dtype = dtype_to_storage_type(tensor.dtype);
        launch_unary_float::<LeakyRelu, _>(tensor, |_| InputScalar::new(negative_slope, dtype))
    }
}
