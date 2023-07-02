use chrono::Local;
use std::{io::BufRead, thread};

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

    let mut camera = Camera::new(index, requested).unwrap();

    // let mut face = opencv::objdetect;
    camera.open_stream().unwrap();

    loop {
        println!("Press enter to take a photo, remember to smile!");
        handle.read_line(&mut input).expect("Failed to read line");

        let frame = camera.frame().unwrap();

        thread::spawn(move || {
            let decoded = frame.decode_image::<RgbFormat>().unwrap();

            let encoded = export::arr_to_image(
                &decoded,
                decoded.width(),
                decoded.height(),
                image::ImageOutputFormat::Png,
            )
            .unwrap();

            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
            std::fs::write(format!("./photo_{timestamp}.png"), encoded).unwrap();
            println!("Finished export");
        });
    }
}
