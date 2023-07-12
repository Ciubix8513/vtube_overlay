use std::io::Cursor;

use image::{ImageBuffer, ImageError, Rgb};

///Transforms an array of raw image bytes into a specified format
pub fn arr_to_image(
    img: &[u8],
    width: u32,
    height: u32,
    format: image::ImageOutputFormat,
) -> Result<Vec<u8>, ImageError> {
    let img = img
        .iter()
        // .chunks_exact(1)
        .map(|i| {
            let mut array = [0; 3];
            // array.copy_from_slice(i);
            array.fill_with(|| *i);
            Rgb(array)
        })
        .collect::<Vec<Rgb<u8>>>();

    let mut image_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width, height);
    let mut x = 0;
    let mut y = 0;

    for i in img {
        let pixel = image_buffer.get_pixel_mut(x, y);
        x += 1;
        if x == width {
            x = 0;
            y += 1;
        }
        *pixel = i;
    }
    let mut byte_stream = Vec::new();
    image_buffer.write_to(&mut Cursor::new(&mut byte_stream), format)?;

    Ok(byte_stream)
}
