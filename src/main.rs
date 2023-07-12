#![allow(clippy::cast_sign_loss)]
use chrono::Local;
use opencv::{
    core::{self, Size},
    imgproc,
    prelude::*,
    videoio::{self, CAP_ANY, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH},
};
use std::{io::BufRead, slice};

mod export;

fn main() {
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut input = String::new();

    let mut capture = videoio::VideoCapture::new(0, CAP_ANY).unwrap();
    capture.set(CAP_PROP_FRAME_WIDTH, 640.0).unwrap();
    capture.set(CAP_PROP_FRAME_HEIGHT, 480.0).unwrap();

    // assert!(!capture.is_opened().unwrap_or_default(), "Camera not open");

    let xml = core::find_file(
        "haarcascades/haarcascade_frontalface_default.xml",
        true,
        false,
    )
    .unwrap();
    let mut face = opencv::objdetect::CascadeClassifier::new(&xml).unwrap();

    loop {
        println!("Press enter to take a photo, remember to smile!");
        handle.read_line(&mut input).expect("Failed to read line");

        let mut gray = Mat::default();
        let mut src = Mat::default();

        capture.read(&mut src).unwrap();
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
            0.25f64,
            0.25f64,
            imgproc::INTER_CUBIC,
        )
        .unwrap();

        println!("Reduced the image to size of {:?}", reduced.size().unwrap());
        let reduced_size = reduced.size().unwrap();
        let data = reduced.data();
        let data_slice = unsafe { slice::from_raw_parts(data, reduced.total()) };

        let encoded = export::arr_to_image(
            data_slice,
            reduced_size.width as u32,
            reduced_size.height as u32,
            image::ImageOutputFormat::Png,
        )
        .unwrap();

        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
        std::fs::write(format!("./debug_photo_{timestamp}.png"), encoded).unwrap();

        println!("Saved debug photo");

        println!("About to run detect multiscale");
        let mut faces = opencv::types::VectorOfRect::new();

        face.detect_multi_scale(
            &reduced,
            &mut faces,
            1.2,
            2,
            0,
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
