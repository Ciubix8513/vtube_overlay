use chrono::Local;
use opencv::{
    core::{self, Size, Vector, CV_8UC3},
    imgproc,
    prelude::*,
};
use std::{io::BufRead, os::raw::c_void, thread};

use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

mod export;

fn main() {
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut input = String::new();

    let index = CameraIndex::Index(0);

    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let xml = core::find_file(
        "haarcascades/haarcascade_frontalface_default.xml",
        true,
        false,
    )
    .unwrap();

    let mut camera = Camera::new(index, requested).unwrap();
    let mut face = opencv::objdetect::CascadeClassifier::new(&xml).unwrap();

    camera.open_stream().unwrap();

    loop {
        println!("Press enter to take a photo, remember to smile!");
        handle.read_line(&mut input).expect("Failed to read line");

        let frame = camera.frame().unwrap();

        let decoded = frame.decode_image::<RgbFormat>().unwrap();

        let mut gray = Mat::default();
        // let src = Mat::from_slice(&decoded.to_vec()).unwrap();

        //Need to have this reference here so that the pointer isn't null decoded.to_vec()
        let mut d_vec = decoded.to_vec();
        let src = unsafe {
            Mat::new_rows_cols_with_data(
                decoded.width() as i32,
                decoded.height() as i32,
                CV_8UC3,
                d_vec.as_mut_ptr() as *mut c_void,
                0,
            )
        }
        .unwrap();

        imgproc::cvt_color(&src, &mut gray, imgproc::COLOR_RGB2GRAY, 0).unwrap();
        println!("Turned image grayscale");

        let mut reduced = Mat::default();

        imgproc::resize(
            &gray,
            &mut reduced,
            core::Size {
                width: 0,
                height: 0,
            },
            0.125f64,
            0.125f64,
            imgproc::INTER_LINEAR,
        )
        .unwrap();

        println!("Reduced the image to size of {:?}", reduced.size().unwrap());
        let mut faces = opencv::types::VectorOfRect::new();
        let reduced_size = reduced.size().unwrap();
        let mut test_output_vec = Vector::<u8>::from_slice(&vec![
            0u8;
            (reduced_size.width * reduced_size.height)
                as usize
        ]);
        reduced
            .copy_to(&mut test_output_vec)
            .expect("Failed to copy into vec");

        let test_output_vec = test_output_vec.to_vec();

        let encoded = export::arr_to_image(
            &test_output_vec,
            reduced_size.width as u32,
            reduced_size.height as u32,
            image::ImageOutputFormat::Png,
        )
        .unwrap();

        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
        std::fs::write(format!("./debug_photo_{timestamp}.png"), encoded).unwrap();

        println!("Saved debug photo");

        println!("About to run detect multiscale");
        face.detect_multi_scale(
            &reduced,
            &mut faces,
            1.1,
            2,
            0,
            // opencv::objdetect::CASCADE_SCALE_IMAGE,
            Size::new(0, 0),
            reduced.size().unwrap(),
        )
        .unwrap();

        println!("Ran detect multiscale");

        println!("Found {} faces in the image", faces.len());

        // let encoded = export::arr_to_image(
        //     &decoded,
        //     decoded.width(),
        //     decoded.height(),
        //     image::ImageOutputFormat::Png,
        // )
        // .unwrap();

        // let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
        // std::fs::write(format!("./photo_{timestamp}.png"), encoded).unwrap();
        // println!("Finished export");
    }
}
