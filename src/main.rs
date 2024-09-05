use anyhow::Result;
use opencv::{
    calib3d::{calibrate_camera, find_chessboard_corners, CALIB_CB_ADAPTIVE_THRESH, CALIB_CB_NORMALIZE_IMAGE},
    core::{Mat, MatTraitConst, Point2f, Point3f, Size, TermCriteria, TermCriteria_Type, Vector, CV_32F},
    highgui::{destroy_all_windows, imshow, wait_key},
    imgproc::{corner_sub_pix, cvt_color, COLOR_BGR2GRAY, COLOR_BGR2RGBA},
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureTrait, VideoWriter, VideoWriterTrait},
};
use slint::{Image, SharedString, Timer, TimerMode};
use std::{
    io::{stderr, Write},
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    thread::{spawn, JoinHandle},
    time::Duration,
};

// Import your Slint UI file
slint::include_modules!();

const CAMERA_INDEX: i32 = 0;

#[derive(Debug)]
enum CalibrationType {
    ChessBoard,
    CircleGrid,
    RabbitPAruco,
}

fn main() -> Result<()> {
    env_logger::init();

    let (frame_sender, frame_receiver) = channel();
    let (exit_sender, exit_receiver) = channel();

    // Wrap frame_receiver in Arc<Mutex<Receiver<T>>>
    let frame_receiver = Arc::new(Mutex::new(frame_receiver));

    // Initialize camera
    let camera = VideoCapture::new(CAMERA_INDEX, videoio::CAP_ANY)?;
    if !camera.is_opened()? {
        panic!("Unable to open default camera!");
    }

    // Get camera parameters
    let frame_width = camera.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = camera.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = camera.get(videoio::CAP_PROP_FPS)?;
    println!(
        "Camera: width {}, height {}, FPS: {}",
        frame_width, frame_height, fps
    );

    // Initialize Slint window
    let window = MainWindow::new()?;
    let window_clone_for_callback = window.as_weak(); // Clone for use in calibration callback
    let window_clone_for_render = window.as_weak(); // Clone for use in render closure

    let frame_receiver_for_callback = Arc::clone(&frame_receiver); // Clone for callback use
    window.on_calibration_wrapper_callback(move |selected_calibration, grid_rows, grid_cols, loc_x, loc_y| {
        // Convert integer to enum
        let calibration_type = match selected_calibration {
            0 => CalibrationType::ChessBoard,
            1 => CalibrationType::CircleGrid,
            2 => CalibrationType::RabbitPAruco,
            _ => {
                eprintln!("Unknown calibration type selected: {}", selected_calibration);
                stderr().flush().unwrap();
                return;
            }
        };

        eprintln!(
            "Calibration started with type: {:?}, rows: {}, cols: {}, loc_x: {}, loc_y: {}",
            calibration_type, grid_rows, grid_cols, loc_x, loc_y
        );
        stderr().flush().unwrap();

        // Perform calibration in a separate thread to avoid blocking the UI
        let window_clone = window_clone_for_callback.clone(); // Clone for use in this thread
        let frame_receiver = Arc::clone(&frame_receiver_for_callback); // Clone again for thread use
        thread::spawn(move || {
            match calibration_type {
                CalibrationType::ChessBoard => {
                    if let Err(e) = start_chessboard_calibration(grid_rows, grid_cols, &frame_receiver, frame_width, frame_height, window_clone) {
                        eprintln!("Error during calibration: {:?}", e);
                    }
                }
                CalibrationType::CircleGrid => {
                    if let Err(e) = start_circle_grid_calibration(grid_rows, grid_cols) {
                        eprintln!("Error during calibration: {:?}", e);
                    }
                }
                CalibrationType::RabbitPAruco => {
                    if let Err(e) = start_aruco_calibration(loc_x, loc_y) {
                        eprintln!("Error during calibration: {:?}", e);
                    }
                }
            }
        });
    });

    // Set up a timer to update frames in the Slint window
    let timer = Timer::default();
    timer.start(
        TimerMode::Repeated,
        Duration::from_secs_f32(1.0 / (fps + 10.0) as f32), // Adjusting for smoother video display
        move || {
            if let Some(window) = window_clone_for_render.upgrade() {
                window.set_frame(window.get_frame() + 1);
            }
        },
    );

    // Start the camera thread to handle capturing frames
    let camera_thread = start_camera_thread(
        frame_sender,
        exit_receiver,
        camera,
        frame_width as f64,
        frame_height as f64,
        fps,
    )?;

    // Use the Arc<Mutex<Receiver>> in the render closure
    let frame_receiver_render = Arc::clone(&frame_receiver);
    let render = move || -> Result<Image> {
        let receiver = frame_receiver_render.lock().unwrap();
        if let Ok(frame_rgba) = receiver.try_recv() {
            let frame_data: Vec<u8> = frame_rgba; // Use directly
            let image = Image::from_rgba8(slint::SharedPixelBuffer::clone_from_slice(
                frame_data.as_slice(),
                frame_width as u32,
                frame_height as u32,
            ));
            Ok(image)
        } else {
            Ok(Image::default())
        }
    };

    // Handle rendering of images in Slint window
    window.on_render_image(move |_frame| {
        render().unwrap_or_else(|err| {
            eprintln!("Error rendering image: {:?}", err);
            Image::default()
        })
    });

    window.run()?;

    exit_sender.send(())?;
    camera_thread.join().unwrap()?;
    println!("Camera stopped and resources released");
    destroy_all_windows()?; // Close all OpenCV windows
    Ok(())
}

fn start_camera_thread(
    frame_sender: Sender<Vec<u8>>,
    exit_receiver: Receiver<()>,
    mut camera: VideoCapture,
    frame_width: f64,
    frame_height: f64,
    fps: f64,
) -> Result<JoinHandle<Result<()>>> {
    Ok(spawn(move || -> Result<()> {
        let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let mut out = VideoWriter::new(
            "output.mp4",
            fourcc,
            fps,
            Size::new(frame_width as i32, frame_height as i32),
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
    }))
}

fn start_chessboard_calibration(
    grid_rows: i32,
    grid_cols: i32,
    frame_receiver: &Arc<Mutex<Receiver<Vec<u8>>>>,
    frame_width: i32,
    frame_height: i32,
    window: slint::Weak<MainWindow>,
) -> Result<()> {
    let board_size = Size::new(grid_cols, grid_rows);

    let object_point_set: Vector<Point3f> = (0..grid_rows)
        .flat_map(|row| (0..grid_cols).map(move |col| Point3f::new(row as f32, col as f32, 0.)))
        .collect();

    let mut captured_frames = 0;
    const REQUIRED_FRAMES: usize = 10; // Number of frames to capture for calibration

    let mut object_points: Vector<Vector<Point3f>> = Vector::new();
    let mut image_points: Vector<Vector<Point2f>> = Vector::new();

    // Capture frames and detect chessboard corners
    while captured_frames < REQUIRED_FRAMES {
        if let Ok(frame_data) = frame_receiver.lock().unwrap().try_recv() {
            let frame_slice = Mat::from_slice(frame_data.as_slice())?;
            let frame_mat = frame_slice.reshape(4, frame_height)?;

            let mut gray = Mat::default();
            cvt_color(&frame_mat, &mut gray, COLOR_BGR2GRAY, 0)?;

            let mut corners = opencv::types::VectorOfPoint2f::new();
            let found = find_chessboard_corners(
                &gray,
                board_size,
                &mut corners,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE,
            )?;

            if found {
                // Refine the corner positions
                let term_criteria = TermCriteria::new(
                    TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
                    30,
                    0.1,
                )?;
                corner_sub_pix(
                    &gray,
                    &mut corners,
                    Size::new(11, 11),
                    Size::new(-1, -1),
                    term_criteria,
                )?;

                image_points.push(corners);
                object_points.push(object_point_set.clone());

                captured_frames += 1;

                // Update status on Slint UI using the generated setter
                if let Some(win) = window.upgrade() {
                    win.set_status(format!("Captured frames: {}", captured_frames).into());
                }
            }

            imshow("Chessboard Calibration", &gray)?;
            if wait_key(1)? == 27 {
                break; // Exit if 'Esc' is pressed
            }
        } else {
            thread::sleep(Duration::from_millis(10));
        }
    }

    // Camera calibration using the captured points
    let mut camera_matrix = Mat::eye(3, 3, CV_32F)?.to_mat()?; // 3x3 camera matrix
    let mut dist_coeffs = Mat::zeros(8, 1, CV_32F)?.to_mat()?; // Distortion coefficients
    let mut rvecs = opencv::types::VectorOfMat::new();
    let mut tvecs = opencv::types::VectorOfMat::new();

    calibrate_camera(
        &object_points,
        &image_points,
        Size::new(frame_width, frame_height),
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        0, // Calibration flags (can be customized)
        TermCriteria::new(
            TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
            30,
            0.1,
        )?,
    )?;

    println!("Camera matrix: {:?}", camera_matrix);
    println!("Distortion coefficients: {:?}", dist_coeffs);

    Ok(())
}

fn start_circle_grid_calibration(grid_rows: i32, grid_cols: i32) -> Result<()> {
    eprintln!(
        "Starting Circle Grid calibration with rows: {}, cols: {}",
        grid_rows, grid_cols
    );
    stderr().flush().unwrap();
    Ok(())
}

fn start_aruco_calibration(loc_x: SharedString, loc_y: SharedString) -> Result<()> {
    eprintln!(
        "Starting Aruco calibration with loc_x: {}, loc_y: {}",
        loc_x, loc_y
    );
    stderr().flush().unwrap();
    Ok(())
}
