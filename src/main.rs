use std::{
    sync::mpsc::{channel, Receiver, Sender},
    thread::{spawn, JoinHandle},
    time::Duration,
};

use anyhow::Result;
use opencv::{
    core::{self, MatTraitConst, Size2i},
    imgproc::{cvt_color, COLOR_BGR2RGBA},
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait, VideoWriter, VideoWriterTrait},
};

use slint::{Image, Timer, TimerMode};

const CAMERA_INDEX: i32 = 0;

// Import your Slint UI file
slint::include_modules!();

fn main() -> Result<()> {
    // Open the camera
    let mut camera = VideoCapture::new(CAMERA_INDEX, videoio::CAP_ANY)?;
    if !camera.is_opened()? {
        panic!("Unable to open default camera!");
    }

    // Get camera parameters
    let frame_width = camera.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let frame_height = camera.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    let fps = camera.get(videoio::CAP_PROP_FPS)?;
    println!(
        "camera: width {}, height {}, FPS: {}",
        frame_width, frame_height, fps
    );

    let window = MainWindow::new().unwrap();  // Use MainWindow here
    let window_clone = window.as_weak();

    let timer = Timer::default();
    timer.start(
        TimerMode::Repeated,
        Duration::from_secs_f32(1. / (fps + 10.0) as f32), // Increase Slint image display frequency for smoother video
        move || {
            if let Some(window) = window_clone.upgrade() {
                window.set_frame(window.get_frame() + 1);
            }
        },
    );

    // Create a channel between Slint and Camera image
    let (frame_sender, frame_receiver) = channel();
    let (exit_sender, exit_receiver) = channel();

    let camera_thread = start_camera_thread(
        frame_sender,
        exit_receiver,
        camera,
        frame_width,
        frame_height,
        fps,
    );

    let mut frame_data = vec![0; (frame_width * frame_height * 4.0) as usize];
    let mut render = move || -> Result<Image> {
        if let Ok(frame_rgba) = frame_receiver.try_recv() {
            frame_data.copy_from_slice(&frame_rgba);
        }
        let image = slint::Image::from_rgba8(slint::SharedPixelBuffer::clone_from_slice(
            frame_data.as_slice(),
            frame_width as u32,
            frame_height as u32,
        ));
        Ok(image)
    };

    window.on_render_image(move |_frame| render().unwrap_or_else(|err| {
        eprintln!("Error rendering image: {:?}", err);
        Image::default()
    }));

    window.run().unwrap();

    exit_sender.send(())?;
    camera_thread.join().unwrap()?;
    println!("Camera stopped and resources released");
    Ok(())
}

fn start_camera_thread(
    frame_sender: Sender<Vec<u8>>,
    exit_receiver: Receiver<()>,
    mut camera: VideoCapture,
    frame_width: f64,
    frame_height: f64,
    fps: f64,
) -> JoinHandle<Result<()>> {
    spawn(move || -> Result<()> {
        let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let mut out = VideoWriter::new(
            "output.mp4",
            fourcc,
            fps,
            Size2i::new(frame_width as i32, frame_height as i32),
            true,
        )?;

        let mut frame_bgr = Mat::default();
        let mut frame_rgba = Mat::default();
        loop {
            if exit_receiver.try_recv().is_ok() {
                break;
            } else {
                camera.read(&mut frame_bgr)?;

                cvt_color(&frame_bgr, &mut frame_rgba, COLOR_BGR2RGBA, 0)?;

                frame_sender.send(frame_rgba.data_bytes()?.to_vec())?;

                if frame_bgr.size()?.width > 0 {
                    out.write(&frame_bgr)?;
                }

                std::thread::sleep(Duration::from_millis(10)); // Add delay to control capture rate
            }
        }
        Ok(())
    })
}
