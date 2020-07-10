#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/image_processing/generic_image.h"
#include "dlib/pixel.h"
#include "dlib/opencv.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "boost/filesystem.hpp"

#include <iostream>
#include <fstream>

void process_obj_images(dlib::frontal_face_detector& frontal_face_detector,
                        const dlib::shape_predictor& face_shape_predictor,
                        const boost::filesystem::path& abs_path_to_processed_data_set_dir,
                        const boost::filesystem::path& abs_path_to_obj_imgs,
                        const std::string& obj_name,
                        const int full_face_shape_size,
                        const double full_face_shape_padding,
                        const bool show_image_processing_stages,
                        const bool draw_rect_around_face
                        )
{

    // create directory for processed images for object.
    {
        const auto dir_name = abs_path_to_processed_data_set_dir.string() + '/' + obj_name;
        if(!boost::filesystem::is_directory(dir_name)) {
            if(!boost::filesystem::create_directory(dir_name)) {
                std::cerr << "Cound not create: " << dir_name << '.'
                          << "We skip this object.\n";
                return;
            }
        }
    }


    boost::filesystem::directory_iterator iter(abs_path_to_obj_imgs);
    boost::filesystem::directory_iterator end;

    std::vector<std::string> abs_imgs_paths;
    std::vector<std::string> imgs_names;
    for(; iter != end; ++iter) {
        abs_imgs_paths.push_back(boost::filesystem::canonical(iter->path()).string());
        imgs_names.push_back(iter->path().filename().string());
    }

    for(std::size_t i = 0; i < abs_imgs_paths.size(); ++i) {

        std::cout << "processing: " << abs_imgs_paths[i] << "...\n";

        dlib::array2d<dlib::bgr_pixel> img;
        dlib::load_image(img, abs_imgs_paths[i]);

        if(show_image_processing_stages) {
            // show loaded image.
            std::cout << "Zero stage\n";
            namedWindow("Zero stage", cv::WINDOW_NORMAL);
            cv::imshow("Zero stage", dlib::toMat(img));
            cv::waitKey(0);
            cv::destroyWindow("Zero stage");
        }

        // first we trying to find exactly one face on the image.

        std::vector<dlib::rectangle> rects_around_faces = frontal_face_detector(img);
        std::cout << "Number of faces detected: " << rects_around_faces.size() << '\n';
        if(rects_around_faces.empty()) {
            std::cout << "Face not found!\n";
            continue;
        }
        if(rects_around_faces.size() > 1) {
            std::cout << "Found more than one face\n";
            continue;
        }
        dlib::rectangle rect_around_full_face = rects_around_faces[0];


        // now we extract face(also align it) and some padding around face from image.

        dlib::matrix<dlib::bgr_pixel> bgr_full_face;
        dlib::full_object_detection full_face_shape = face_shape_predictor(img, rect_around_full_face);
        dlib::extract_image_chip(img,
                                 dlib::get_face_chip_details(full_face_shape, full_face_shape_size, full_face_shape_padding),
                                 bgr_full_face);

        if(show_image_processing_stages) {
            // show extracted and aligned face and some padding around face.
            std::cout << "First stage\n";
            namedWindow("First stage", cv::WINDOW_NORMAL);
            cv::imshow("First stage", dlib::toMat(bgr_full_face));
            cv::waitKey(0);
            cv::destroyWindow("First stage");
        }


        // now we again extract face and additionally 68 points on the face.

        std::vector<dlib::rectangle> rects_around_little_face = frontal_face_detector(bgr_full_face);
        if(rects_around_little_face.size() != 1) {
            std::cerr << "Could not find little face. Skip this image.\n";
            continue;
        }

        dlib::rectangle rect_around_little_face = rects_around_little_face[0];
        dlib::full_object_detection little_face_shape = face_shape_predictor(bgr_full_face, rect_around_little_face);

        std::vector<dlib::point> little_face_points;
        const auto number_of_points = little_face_shape.num_parts();
        for(std::size_t j = 0; j < number_of_points; ++j) {
            little_face_points.push_back(little_face_shape.part(j));
        }


        // crucial face points. Later we will crop face around these points.

        // points near ears.
        dlib::point point_0 = little_face_points[0];
        dlib::point point_1 = little_face_points[16];

        // points under mouth.
        dlib::point point_2 = little_face_points[5];
        dlib::point point_3 = little_face_points[11];

        // points above the eyes.
        dlib::point point_4 = little_face_points[19];
        dlib::point point_5 = little_face_points[24];


        // create rectangle from crucial points.

        dlib::point bl(little_face_points[4]);
        dlib::point br(little_face_points[12]);

        // max y?
        int max_y = std::max(bl.y(), br.y());
        bl.y() = max_y;
        br.y() = max_y;

        dlib::point tl(bl.x(), point_4.y());
        dlib::point tr(br.x(), point_5.y());

        // min y?
        int min_y = std::min(tl.y(), tr.y());
        tl.y() = min_y;
        tr.y() = min_y;

        if(draw_rect_around_face) {
            dlib::draw_line(bgr_full_face, bl, br, dlib::rgb_pixel{0,255,255});
            dlib::draw_line(bgr_full_face, br, tr, dlib::rgb_pixel{0,255,255});
            dlib::draw_line(bgr_full_face, tr, tl, dlib::rgb_pixel{0,255,255});
            dlib::draw_line(bgr_full_face, tl, bl, dlib::rgb_pixel{0,255,255});
        }


        if(show_image_processing_stages) {
            // show rectangle around face. (If you set variable draw_rect_around_face to false this stage will looks like previous stage).
            std::cout << "Second stage\n";
            namedWindow("Second stage", cv::WINDOW_NORMAL);
            cv::imshow("Second stage", dlib::toMat(bgr_full_face));
            cv::waitKey(0);
            cv::destroyWindow("Second stage");
        }

        // now we resize face.
        dlib::rectangle dlib_processed_face_rect(tl, br);
        cv::Rect cv_processed_face_rect(dlib_processed_face_rect.left(), dlib_processed_face_rect.top(),
                                        dlib_processed_face_rect.width(), dlib_processed_face_rect.height());

        cv::Mat processed_face = dlib::toMat(bgr_full_face)(cv_processed_face_rect);

        // all faces must have the same size. (It is necessary for face recognition algorithm).
        static int width = processed_face.cols;
        static int height = processed_face.rows;

        cv::Mat resized_processed_face;
        cv::resize(processed_face, resized_processed_face, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);

        if(show_image_processing_stages) {
            // show resized face.
            std::cout << "Third stage\n";
            namedWindow("Third stage", cv::WINDOW_NORMAL);
            cv::imshow("Third stage", resized_processed_face);
            cv::waitKey(0);
            cv::destroyWindow("Third stage");
        }


        // now we convert face to gray.

        dlib::matrix<unsigned char> gray_processed_face;
        dlib::assign_image(gray_processed_face, dlib::cv_image<dlib::bgr_pixel>(resized_processed_face));

        if(show_image_processing_stages) {
            // show fully processed face.
            std::cout << "Fourth stage\n";
            namedWindow("Fourth stage", cv::WINDOW_NORMAL);
            cv::imshow("Fourth stage", dlib::toMat(gray_processed_face));
            cv::waitKey(0);
            cv::destroyWindow("Fourth stage");
        }

        // save processed face.
        auto processed_img_path = abs_path_to_processed_data_set_dir.string() + '/' + obj_name + '/' + imgs_names[i];
        std::cout << "save processed img in: " << processed_img_path << '\n';
        dlib::save_jpeg(gray_processed_face, processed_img_path);
    }
}

int main(int argc, const char** argv)
{
    try
    {
        if(argc != 8) {
            std::cerr << "Usage error.\n"
                      << "Usage:\n"
                      << "[1] <path_to_face_landmarks.dat>\n"
                      << "[2] <path_to_data_set_dir>\n"
                      << "[3] <path_to_processed_data_set_dir> (directory must exists!)\n"
                      << "[4] <full_face_shape_size> (int > 0)\n"
                      << "[5] <full_face_shape_padding> (double >= 0)\n"
                      << "[6] <show image processing stages?> (int: 0 -> false, 1 -> true)\n"
                      << "[7] <draw rect around face?> (int: 0 -> false, 1 -> true)\n";
            return -1;
        }

        // parse command line arguments.

        const std::string path_to_face_landmarks(argv[1]);
        const std::string path_to_data_set_dir(argv[2]);
        const auto abs_path_to_processed_data_set_dir = boost::filesystem::canonical(argv[3]);
        const int full_face_shape_size = std::stoi(argv[4]);
        const double full_face_shape_padding = std::stod(argv[5]);

        const int show_image_processing_stages_temp = std::stoi(argv[6]);
        const bool show_image_processing_stages = show_image_processing_stages_temp == 0 ? false : true;

        const int draw_rect_around_face_temp = std::stol(argv[7]);
        const bool draw_rect_around_face = draw_rect_around_face_temp == 0 ? false : true;

        std::cout << "we will save processed images in " << abs_path_to_processed_data_set_dir.string() << '\n';


        // create frontal face detector and shape predictor.

        dlib::frontal_face_detector frontal_face_detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor face_shape_predictor;
        dlib::deserialize(path_to_face_landmarks) >> face_shape_predictor;


        // Process every dir in data_set_dir. (one directory per object).

        boost::filesystem::directory_iterator data_set_dir_iter(path_to_data_set_dir);
        boost::filesystem::directory_iterator end;

        for(; data_set_dir_iter != end; ++data_set_dir_iter) {

            // process only directories
            const auto file_status = boost::filesystem::status(*data_set_dir_iter);

            if(file_status.type() == boost::filesystem::directory_file) {

                const auto abs_path_to_obj_imgs = boost::filesystem::canonical(*data_set_dir_iter);
                const auto obj_name = data_set_dir_iter->path().filename().string();

                std::cout << "process images in " << abs_path_to_obj_imgs << '\n';
                std::cout << "\n************************************\n";
                process_obj_images(frontal_face_detector, face_shape_predictor,
                                   abs_path_to_processed_data_set_dir,
                                   abs_path_to_obj_imgs,
                                   obj_name,
                                   full_face_shape_size,
                                   full_face_shape_padding,
                                   show_image_processing_stages,
                                   draw_rect_around_face);
                std::cout << "\n************************************\n";
            }

        }

    }
    catch (const std::exception& e)
    {
        std::cout << "\nexception thrown!\n";
        std::cout << e.what() << '\n';
        return -1;
    }

    return 0;
}
