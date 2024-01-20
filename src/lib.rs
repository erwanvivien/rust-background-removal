use image::{imageops, DynamicImage, ImageBuffer};

use ndarray::{Array, CowArray};
use ort::{OrtError, Value};

mod onnx;

pub fn process_image(
    input_image: &DynamicImage,
) -> Result<ImageBuffer<image::Rgba<u8>, Vec<u8>>, OrtError> {
    static mut SESSION: Option<ort::InMemorySession> = None;

    if unsafe { SESSION.is_none() } {
        unsafe { SESSION = Some(onnx::onnx_session()?) };
    }
    let session = unsafe { SESSION.as_ref().unwrap() };

    let input_shape = session.inputs[0]
        .dimensions()
        .map(|dim| dim.unwrap())
        .collect::<Vec<usize>>();

    let input_image = input_image.to_rgba8();
    let scaling_factor = f32::min(
        input_shape[3] as f32 / input_image.width() as f32, // Width ratio
        input_shape[2] as f32 / input_image.height() as f32, // Height ratio
    )
    .min(1.0); // Don't upscale

    let mut resized_image = imageops::resize(
        &input_image,
        input_shape[3] as u32,
        input_shape[2] as u32,
        imageops::FilterType::Triangle,
    );

    let mean = 128.;
    let std = 256.;
    let input_tensor = CowArray::from(
        Array::from_shape_fn(input_shape, |indices| {
            (resized_image[(indices[3] as u32, indices[2] as u32)][indices[1]] as f32 - mean) / std
        })
        .into_dyn(),
    );

    let inputs = vec![Value::from_array(session.allocator(), &input_tensor)?];
    let outputs = session.run(inputs)?;
    let output_tensor = outputs[0].try_extract::<f32>()?;
    for (indices, alpha) in output_tensor.view().indexed_iter() {
        resized_image[(indices[3] as u32, indices[2] as u32)][3] = (alpha * 255.) as u8;
    }
    let output_image = imageops::resize(
        &resized_image,
        (input_image.width() as f32 * scaling_factor) as u32,
        (input_image.height() as f32 * scaling_factor) as u32,
        imageops::FilterType::Triangle,
    );

    Ok(output_image)
}
