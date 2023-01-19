// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

//Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform)
{
    //TODO: setup uniform
    uniform.view_trafo <<1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;
    uniform.view_trafo(1,1)=aspect_ratio;
    //TODO: setup camera, compute w, u, v
    
    double t = near_plane*(tan(field_of_view/2)); 
    double r = t * aspect_ratio; 
    //is_perspective=false;
    
    Vector3d w = -(camera_gaze/camera_gaze.norm());
    Vector3d u = (camera_top.cross(w))/((camera_top.cross(w)).norm());
    Vector3d v = w.cross(u);
    //TODO: compute the camera transformation
    Matrix4f ctm;
    ctm <<u[0],v[0],w[0],camera_position[0],u[1],v[1],w[1],camera_position[1],u[2],v[2],w[2],camera_position[2],0,0,0,1;
    //ctm=pow(ctm,-1);
    //TODO: setup projection matrix
    double n=-near_plane;
    double f=-far_plane;
    double b= -near_plane*(tan(field_of_view/2)); 
    double l= b * aspect_ratio; 
    Matrix4f P;
    // P <<2/(r-l),0,0,-(r+l)/(r-l),0,2/(t-b),0,-(t+b)/(t-b),0,0,2/(n-f),-(n+f)/(n-f),0,0,0,1;
    if (is_perspective)
    {
        P << (2*n)/(r-l), 0, (r+l)/(r-l), 0, 0, (2*n)/(t-b), (t+b)/(t-b), 0, 0, 0, -(n+f)/(n-f), (-2*f*n)/(f-n), 0, 0, 1, 0;
        //TODO setup prespective camera
    }
    else
    {
        P <<2/(r-l),0,0,-(r+l)/(r-l),0,2/(t-b),0,-(t+b)/(t-b),0,0,2/(n-f),-(n+f)/(n-f),0,0,0,1;
    }
    uniform.view_trafo<<P*ctm.inverse();
    //uniform.view_trafo<<P*ctm;
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        VertexAttributes outputret=va;
        outputret.position=uniform.view_trafo*outputret.position;
        return outputret;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1,0,0);
        
        
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        // double alpha = fa.color(3);
        // Eigen::Vector3d new_color(fa.color(0),fa.color(1),fa.color(2));
        // Eigen::Vector3d previous_color(previous.color(0)/255.0,previous.color(1)/255.0,previous.color(2)/255.0);
        // Eigen::Vector3d out_color = (1-alpha)*previous_color+alpha*new_color;
        // if(fa.depth<previous.depth){
        //     FrameBufferAttributes out(out_color[0]*255,out_color[1]*255,out_color[2]*255,255);
        //     out.depth=fa.depth;
        //     return out;
        // }else{
        //     return previous;
        // }
        //return FrameBufferAttributes(obj_diffuse_color(0),obj_diffuse_color(1),obj_diffuse_color(2),255);
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    //TODO: build the vertex attributes from vertices and facets
    vertex_attributes.clear();
    //uniform.color<<0,1,0,1;
    for (int i = 0; i < facets.size()/3; i++) {
        const Vector3d A1 = vertices.row(facets(i,0)).transpose();
        const Vector3d B1 = vertices.row(facets(i,1)).transpose();
        const Vector3d C1 = vertices.row(facets(i,2)).transpose();
        vertex_attributes.push_back(VertexAttributes(A1(0),A1(1),A1(2)));
        vertex_attributes.push_back(VertexAttributes(B1(0),B1(1),B1(2)));
        vertex_attributes.push_back(VertexAttributes(C1(0),C1(1),C1(2)));
    }
    
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
    
}

Matrix4f compute_rotation(const double alpha)
{
    //TODO: Compute the rotation matrix of angle alpha on the y axis around the object barycenter
    Matrix4f res;
    res<<cos(alpha),0,sin(alpha),0,0,1,0,0,-sin(alpha),0,cos(alpha),0,0,0,0,1;

    return res;
}

void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4f trafo = compute_rotation(alpha);
    uniform.view_trafo*=trafo;
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        VertexAttributes outputret=va;
        outputret.position=uniform.view_trafo*outputret.position;
        return outputret;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: generate the vertex attributes for the edges and rasterize the lines
    //TODO: use the transformation matrix
    vertex_attributes.clear();
    //uniform.color<<0,1,0,1;
    for (int i = 0; i < facets.size()/3; i++) {
        const Vector3d A1 = vertices.row(facets(i,0)).transpose();
        const Vector3d B1 = vertices.row(facets(i,1)).transpose();
        const Vector3d C1 = vertices.row(facets(i,2)).transpose();
        // Vector3d AB1=A1+B1;
        // Vector3d AC1=A1+C1;
        // Vector3d BC1=B1+C1;
        vertex_attributes.push_back(VertexAttributes(A1(0),A1(1),A1(2)));
        vertex_attributes.push_back(VertexAttributes(B1(0),B1(1),B1(2)));
        vertex_attributes.push_back(VertexAttributes(A1(0),A1(1),A1(2)));
        vertex_attributes.push_back(VertexAttributes(C1(0),C1(1),C1(2)));
        vertex_attributes.push_back(VertexAttributes(B1(0),B1(1),B1(2)));
        vertex_attributes.push_back(VertexAttributes(C1(0),C1(1),C1(2)));
    }
    
    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}

// void gif_wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
// {
    
    
// }


void get_shading_program(Program &program)
{
    
    
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        VertexAttributes out = va;
        out.position = uniform.view_trafo * out.position;
        Vector3d p(va.position(0),va.position(1),va.position(2));
        //error?
        Vector3d N=va.normal;
        Vector3d lights_color(0,0,0);
        for (int i = 0; i < light_positions.size(); ++i)
        {
            const Vector3d &light_position = light_positions[i];
            const Vector3d &light_color = light_colors[i];
            //cout<<light_color;
            Vector3d diff_color = obj_diffuse_color;
            //cout<<diff_color;
            const Vector3d Li = (light_position - p).normalized();
            const Vector3d diffuse = diff_color * std::max(Li.dot(N), 0.0);
            const Vector3d Hi = (Li - (p - camera_position)).normalized();
            const Vector3d specular = obj_specular_color * std::pow(std::max(N.dot(Hi), 0.0), obj_specular_exponent);
            //cout<<Hi;
            const Vector3d D = light_position - p;
            lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
            //cout<<lights_color;
        }
        //cout<<normal<<"\n\n";
        Vector3d C =ambient_light+lights_color;
        out.color=C;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: create the correct fragment
        FragmentAttributes out(va.color(0), va.color(1), va.color(2));
        out.depth = -va.position(2);
        return out;
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: implement the depth check
        //return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
        double alpha = fa.color(3);
        Eigen::Vector3d new_color(fa.color(0), fa.color(1), fa.color(2));
        Eigen::Vector3d previous_color(previous.color(0) / 255.0, previous.color(1) / 255.0, previous.color(2) / 255.0);
        Eigen::Vector3d out_color = new_color;
        if (fa.depth < previous.depth) {
            FrameBufferAttributes out(out_color[0] * 255, out_color[1] * 255, out_color[2] * 255, 255);
            out.depth = fa.depth;
            return out;
        } else {
            return previous;
        }
    };
}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    Eigen::Matrix4f trafo = compute_rotation(alpha);
    uniform.view_trafo*=trafo;
    std::vector<VertexAttributes> vertex_attributes;
    //TODO: compute the normals
    //TODO: set material colors
    vertex_attributes.clear();
    //uniform.color<<0,1,0,1;
    for (int i = 0; i < facets.size()/3; i++) {
        //initial
        const Vector3d A1 = vertices.row(facets(i,0));
        const Vector3d B1 = vertices.row(facets(i,1));
        const Vector3d C1 = vertices.row(facets(i,2));
        //new VertexAttributes
        VertexAttributes A2(A1(0),A1(1),A1(2));
        VertexAttributes B2(B1(0),B1(1),B1(2));
        VertexAttributes C2(C1(0),C1(1),C1(2));
        //error?
        //normalchange
        A2.normal=(C1-A1).cross(C1-B1).normalized();
        B2.normal=(C1-A1).cross(C1-B1).normalized();
        C2.normal=(C1-A1).cross(C1-B1).normalized();
        //pushback
        vertex_attributes.push_back(A2);
        vertex_attributes.push_back(B2);
        vertex_attributes.push_back(C2);
    }
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    
    Eigen::Matrix4f trafo = compute_rotation(alpha);
    uniform.view_trafo*=trafo;
    //TODO: compute the vertex normals as vertex normal average

    std::vector<VertexAttributes> vertex_attributes;
    //TODO: create vertex attributes
    //TODO: set material colors
    MatrixXd pvshading;
    pvshading.resize(vertices.rows(),3);
    pvshading.setZero();
    int counting;
    //count<<pvshading;
    for (int i = 0; i < facets.size()/3; i++) {
        const Vector3d A1=vertices.row(facets(i,0));
        const Vector3d B1=vertices.row(facets(i,1));
        const Vector3d C1=vertices.row(facets(i,2));
        Vector3d pvshadingnormal = (C1-A1).cross(C1-B1).normalized();
        // pvshading(A1(0))+=pvshadingnormal(0);
        // pvshading(B1(0))+=pvshadingnormal(0);
        // pvshading(C1(0))+=pvshadingnormal(0);
        // pvshading(A1(1))+=pvshadingnormal(1);
        // pvshading(B1(1))+=pvshadingnormal(1);
        // //cout<<pvshadingnormal;
        // pvshading(C1(1))+=pvshadingnormal(1);
        // pvshading(A1(2))+=pvshadingnormal(2);
        // pvshading(B1(2))+=pvshadingnormal(2);
        // pvshading(C1(2))+=pvshadingnormal(2);
        pvshading.row(facets(i,0)) += pvshadingnormal;
        pvshading.row(facets(i,1)) += pvshadingnormal;
        pvshading.row(facets(i,2)) += pvshadingnormal;
        counting++;
        //cout<<counting;
    }


    for (int i = 0; i < facets.size()/3; i++) {
        const Vector3d A1=vertices.row(facets(i,0));
        const Vector3d B1=vertices.row(facets(i,1));
        const Vector3d C1=vertices.row(facets(i,2));
        VertexAttributes A2(A1(0),A1(1),A1(2));
        VertexAttributes B2(B1(0),B1(1),B1(2));
        VertexAttributes C2(C1(0),C1(1),C1(2));

        // pvshading(A1(0))/=counting;
        // pvshading(A1(1))/=counting;
        // pvshading(A1(2))/=counting;
        
        // pvshading(B1(0))/=counting;
        // pvshading(B1(1))/=counting;
        // pvshading(B1(2))/=counting;

        // pvshading(C1(0))/=counting;
        // pvshading(C1(1))/=counting;
        // pvshading(C1(2))/=counting;
        pvshading.row(facets(i,0)) /= counting;
        pvshading.row(facets(i,1)) /= counting;
        pvshading.row(facets(i,2)) /= counting;

        A2.normal = pvshading.row(facets(i,0)).normalized();
        B2.normal = pvshading.row(facets(i,1)).normalized();
        C2.normal = pvshading.row(facets(i,2)).normalized();
        // A2.normal(0)=pvshading(A1(0));
        // A2.normal(1)=pvshading(A1(1));
        // A2.normal(2)=pvshading(A1(2));
        // //cout<<A2.normal;
        // B2.normal(0)=pvshading(B1(0));
        // B2.normal(1)=pvshading(B1(1));
        // B2.normal(2)=pvshading(B1(2));
        // //cout<<B2.normal;
        // C2.normal(0)=pvshading(C1(0));
        // C2.normal(1)=pvshading(C1(1));
        // C2.normal(2)=pvshading(C1(2));
        // //cout<<C2.normal; 
        vertex_attributes.push_back(A2);
        //cout<<A2;
        vertex_attributes.push_back(B2);
        //cout<<B2;
        vertex_attributes.push_back(C2);
        //cout<<C2;
    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}


int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());

    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());

    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());

    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());
    //TODO: add the animation
    
    
    {
    GifWriter g;
    GifBegin(&g , "wireframe.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    //GifBegin(&g ,"flat.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    //uniform.color<<0,1,0,1;
    for(double t=0; t<2*EIGEN_PI;t+=1){
        frameBuffer.setConstant(FrameBufferAttributes(0,0,0));
        wireframe_render(t, frameBuffer);
        //flat_render(t, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(),frameBuffer.rows(), frameBuffer.cols(), 25);
    }
    GifEnd(&g);
    }
    {
    GifWriter g;
    GifBegin(&g , "vertex.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    //GifBegin(&g ,"flat.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    //uniform.color<<0,1,0,1;
    for(double t=0; t<2*EIGEN_PI;t+=1){
        frameBuffer.setConstant(FrameBufferAttributes(0,0,0));
        pv_shading(t, frameBuffer);
        //flat_render(t, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(),frameBuffer.rows(), frameBuffer.cols(), 25);
    }
    GifEnd(&g);
    }
    {
    GifWriter g;
    //GifBegin(&g , "wireframe.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    GifBegin(&g ,"flat.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    //uniform.color<<0,1,0,1;
    for(double t=0; t<2*EIGEN_PI;t+=1){
        frameBuffer.setConstant(FrameBufferAttributes(0,0,0));
        //wireframe_render(t, frameBuffer);
        flat_shading(t, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(),frameBuffer.rows(), frameBuffer.cols(), 25);
    }
    GifEnd(&g);
    }
    
    
    // GifBegin(&g ,"flat.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    // frameBuffer.setConstant(FrameBufferAttributes());


    // GifBegin(&g ,"vertex.gif", frameBuffer.rows(), frameBuffer.cols(), 25);
    // frameBuffer.setConstant(FrameBufferAttributes());

    return 0;
}
