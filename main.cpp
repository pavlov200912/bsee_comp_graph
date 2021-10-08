#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include<set>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <unordered_map>

std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
        R"(#version 330 core
uniform mat4 view;
uniform mat4 transform;
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;
out vec4 color;
void main()
{
	gl_Position = view * transform * vec4(in_position, 1.0);
	color = in_color;
}
)";

const char fragment_shader_source[] =
        R"(#version 330 core
in vec4 color;
layout (location = 0) out vec4 out_color;
void main()
{
	out_color = color;
}
)";

GLuint create_shader(GLenum type, const char * source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

struct vec3
{
    float x;
    float y;
    float z;
};

struct vertex
{
    vec3 position;
    std::uint8_t color[4];
};


static vertex cube_vertices[]
        {
                // -X
                {{-1.f, -1.f, -1.f}, {  0, 255, 255, 255}},
                {{-1.f, -1.f,  1.f}, {  0, 255, 255, 255}},
                {{-1.f,  1.f, -1.f}, {  0, 255, 255, 255}},
                {{-1.f,  1.f,  1.f}, {  0, 255, 255, 255}},
                // +X
                {{ 1.f, -1.f,  1.f}, {255,   0,   0, 255}},
                {{ 1.f, -1.f, -1.f}, {255,   0,   0, 255}},
                {{ 1.f,  1.f,  1.f}, {255,   0,   0, 255}},
                {{ 1.f,  1.f, -1.f}, {255,   0,   0, 255}},
                // -Y
                {{-1.f, -1.f, -1.f}, {255,   0, 255, 255}},
                {{ 1.f, -1.f, -1.f}, {255,   0, 255, 255}},
                {{-1.f, -1.f,  1.f}, {255,   0, 255, 255}},
                {{ 1.f, -1.f,  1.f}, {255,   0, 255, 255}},
                // +Y
                {{-1.f,  1.f,  1.f}, {  0, 255,   0, 255}},
                {{ 1.f,  1.f,  1.f}, {  0, 255,   0, 255}},
                {{-1.f,  1.f, -1.f}, {  0, 255,   0, 255}},
                {{ 1.f,  1.f, -1.f}, {  0, 255,   0, 255}},
                // -Z
                {{ 1.f, -1.f, -1.f}, {255, 255,   0, 255}},
                {{-1.f, -1.f, -1.f}, {255, 255,   0, 255}},
                {{ 1.f,  1.f, -1.f}, {255, 255,   0, 255}},
                {{-1.f,  1.f, -1.f}, {255, 255,   0, 255}},
                // +Z
                {{-1.f, -1.f,  1.f}, {  0,   0, 255, 255}},
                {{ 1.f, -1.f,  1.f}, {  0,   0, 255, 255}},
                {{-1.f,  1.f,  1.f}, {  0,   0, 255, 255}},
                {{ 1.f,  1.f,  1.f}, {  0,   0, 255, 255}},
        };

static std::uint32_t cube_indices[]
        {
                // -X
                0, 1, 2, 2, 1, 3,
                // +X
                4, 5, 6, 6, 5, 7,
                // -Y
                8, 9, 10, 10, 9, 11,
                // +Y
                12, 13, 14, 14, 13, 15,
                // -Z
                16, 17, 18, 18, 17, 19,
                // +Z
                20, 21, 22, 22, 21, 23,
        };

// Square matrix multiplication
void matrix_multiply(const float* a, const float* b, float* c, size_t size) {
    std::fill(c, c + size * size, 0);
    for (int i = 0; i < size; i ++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                // c[i, j] = a[i, k] * b[k, j]
                c[i * size + j] += a[i * size + k] * b[k * size + j];
//                std::cout << "pos " << i * size + j << ' ' << i * size + k << '*' << k * size + j << '\n';
//                std::cout << "val " << c[i * size + j] << ' ' << a[i * size + k] << '*' << b[k * size + j] << "\n \n";
            }
        }
    }
}


//float function(int x, int y, GridSize size) {
//    float x_point = // TODO: [0, GridSize.x] -> [-3, 3];
//    float y_point = ;
//    return // TODO: Function f(x, y)
//}

//std::uint32_t get_vertex_index(int x, int y, int z, GridSize size) {
//    // [0..a]x[0..b]x[0..c] --> [0..a*b*c]
//    return x + size.x * (z + size.y * y);
//}
//
//std::tuple<int, int, int> index_to_vertex(int index, GridSize size) {
//    int y = index / (size.x * size.y);
//    int z = (index / (size.x)) % size.x;
//    int x = index % size.x;
//    return {x, y, z};
//}
//
float grid_coordinate_to_float(int x, int grid_size) {
    return (float)x / grid_size;
}

int grid_vertex_to_int(int x, int y, int z, int grid_size) {
    return x + grid_size * (z + grid_size * y);
}

std::tuple<int, int, int> int_to_grid_vertex(int index, int grid_size) {
    int y = index / (grid_size * grid_size);
    int z = (index / grid_size) % grid_size;
    int x = index % grid_size;
    return {x, y, z};
}

vertex set_up_grid_vertex(int x, int y, int z, int grid_size, int index, int vertex_int,
                        const std::set<std::uint32_t> &grid_corners) {

    vertex new_vertex = {{
                                 grid_coordinate_to_float(x, grid_size),
                                 grid_coordinate_to_float(y, grid_size),
                                   grid_coordinate_to_float(z, grid_size)},
                         {50, 100, 100, 0}};
    if (grid_corners.contains(vertex_int)) {
        new_vertex.color[0] = 0;
        new_vertex.color[1] = 0;
        new_vertex.color[2] = 0;
        new_vertex.color[3] = 0;
    }
    return new_vertex;
}

void push_grid_vertex(int x, int y, int z, int grid_size,
                      std::vector<vertex> &vertices, std::unordered_map<int, int> &index_map,
                      const std::set<std::uint32_t> &corners) {
    int index = vertices.size();
    int vertex_int = grid_vertex_to_int(x, y, z, grid_size);
    auto v = set_up_grid_vertex(x, y, z, grid_size, index, vertex_int, corners);
    index_map[vertex_int] = index;
    vertices.push_back(v);
}

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 4",
                                           SDL_WINDOWPOS_CENTERED,
                                           SDL_WINDOWPOS_CENTERED,
                                           800, 600,
                                           SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint transform_location = glGetUniformLocation(program, "transform");

    // Set up gridSize
    int grid_size = 10;
    // Init graph vertices
    std::unordered_map<int, int> grid_vertex_to_index;
    std::vector<vertex> grid_vertices;

    // Set of corner vertex, to color it black
    std::set<std::uint32_t> grid_corners;
    grid_corners.insert(grid_vertex_to_int(0, 0, 0, grid_size));
    grid_corners.insert(grid_vertex_to_int(grid_size - 1, 0, 0, grid_size));
    grid_corners.insert(grid_vertex_to_int(0, grid_size - 1, 0, grid_size));
    grid_corners.insert(grid_vertex_to_int(0, 0, grid_size - 1, grid_size));
    grid_corners.insert(grid_vertex_to_int(grid_size - 1, grid_size - 1, 0, grid_size));
    grid_corners.insert(grid_vertex_to_int(0, grid_size - 1, grid_size - 1, grid_size));
    grid_corners.insert(grid_vertex_to_int(grid_size - 1, 0, grid_size - 1, grid_size));


    for (int i = 0; i < grid_size; i++) {
        // X plane grid vertices
        push_grid_vertex(i, 0, 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(i, 0, grid_size - 1, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);

        push_grid_vertex(0, 0, i, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(grid_size - 1, 0, i, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);

        // Z plane
        push_grid_vertex(0, i , 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(0, i , grid_size - 1, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(0, grid_size - 1 , i, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);

        // Y plane
        push_grid_vertex(i, grid_size - 1 , 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(grid_size - 1, i , 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
    }

//    std::cout << "Grid vertex:" << '\n';
//    for (int i = 0; i < grid_vertices.size(); i++) {
//        auto v = grid_vertices[i].position;
//        auto pos = grid_vertex_to_int(int(v.x * grid_size), int(v.y * grid_size), int(v.z * grid_size), grid_size);
//        std::cout << "Index: " << i << " Vertex: " << int(v.x * grid_size) << ' ' << int(v.y * grid_size) << ' ' << int(v.z * grid_size) << " | " <<
//                                       pos << " | " << grid_vertex_to_index[pos] << '\n';
//    }


    GLuint vbo;
    glGenBuffers(1, &vbo);


    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 grid_vertices.size() * sizeof(grid_vertices[0]),
                 grid_vertices.data(), GL_STATIC_DRAW);

    GLuint vao;

    glGenVertexArrays(1, &vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindVertexArray(vao);


    std::vector<std::uint32_t> line_indices;

    // MAIN GRID
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(grid_size - 1, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, grid_size - 1, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, grid_size - 1, grid_size)]);

    // X grid

    for (int i = 0; i < grid_size; i++) {
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, 0, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, 0, grid_size - 1, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, 0, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, grid_size - 1, 0, grid_size)]);

        std::cout << grid_vertex_to_int(0, i, 0, grid_size) << '\n';
        std::cout << grid_vertex_to_index[grid_vertex_to_int(0, i, 0, grid_size)] << '\n';
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, i, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(grid_size - 1, i, 0, grid_size)]);
//
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, i, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, i, grid_size - 1, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, i, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(grid_size - 1, 0, i, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, i, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, grid_size - 1, i, grid_size)]);
    }

    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, line_indices.size() * sizeof(line_indices[0]), line_indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), reinterpret_cast<void*>(offsetof(vertex, position)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), reinterpret_cast<void*>(offsetof(vertex, color)));



    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_CULL_FACE);
//    glCullFace(GL_FRONT);

    bool running = true;

    float cube_x = 0;
    float cube_y = 0;
    float dt = 0.f;
    float speed = 2.f;
    float z_shift = 1.5;

    float y_angle = 5.45;
    float x_angle = -0.5;

    float scale = 0.5f;

    while (running)
    {

        glClear(GL_DEPTH_BUFFER_BIT);

        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
            {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT: switch (event.window.event)
                    {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            glViewport(0, 0, width, height);
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_LEFT)
                    {
                        cube_x -= speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_RIGHT)
                    {
                        cube_x += speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_UP) {
                        cube_y += speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_DOWN) {
                        cube_y -= speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_s) {
//                        z_shift += speed * dt;
                        x_angle += speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_w) {
//                        z_shift -= speed * dt;
                        x_angle -= speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_d) {
                        y_angle -= speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_a) {
                        y_angle += speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_SPACE) {
                        scale += speed * dt;
                    }
                    else if (event.key.keysym.sym == SDLK_LSHIFT) {
                        scale -= speed * dt;
                    }
                    break;
                case SDL_KEYUP:
                    button_down[event.key.keysym.sym] = false;
                    break;
            }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        float near = 0.1f;
        float far = 10.0;
        float right = near * tan(M_PI / 4.f);
        float top = right * (height / (float) width);
        float left = -right;
        float bottom = -top;


        glClear(GL_COLOR_BUFFER_BIT);

//        Draw 3 cubes
        float view[16] =
                {
                        2 * near / (right - left), 0.f, (right + left) / (right - left), 0.f,
                        0.f, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0.f,
                        0.f, 0.f, - (far + near) / (far - near), - (2 * far * near) / (far - near),
                        0.f, 0.f, -1.f, 0.f,
                };

//        scale = 0.5;

        float scale_matrix[16] = {
                scale, 0, 0, 0,
                0, scale, 0, 0,
                0, 0, scale, 0,
                0, 0, 0, 1.f
        };

        float x_rotate[16] = {
                1.f, 0, 0, 0,
                0, cos(x_angle), sin(x_angle), 0,
                0, -sin(x_angle), cos(x_angle), 0,
                0, 0, 0, 1
        };

        float y_rotate[16] = {
                cos(y_angle), 0, sin(y_angle), 0,
                0, 1, 0, 0,
                -sin(y_angle), 0, cos(y_angle), 0,
                0, 0, 0, 1
        };

        float shift_matrix[16] = {
                1, 0, 0, cube_x,
                0, 1, 0, cube_y,
                0, 0, 1, -z_shift,
                0, 0, 0, 1,
        };

        float transform[16];
        float scale_shift[16];
        float rotate[16];
//        matrix_multiply(scale_matrix, x_rotate, scale_shift, 4);

        matrix_multiply(shift_matrix, scale_matrix, scale_shift, 4);
        matrix_multiply(x_rotate, y_rotate, rotate, 4);
        matrix_multiply(scale_shift, rotate, transform, 4);

//        for (int i = 0; i < 4; i ++) {
//            for (int j = 0; j < 4; j++) {
//                std::cout << transform[i * 4 + j] << ' ';
//            }
//            std::cout << '\n';
//        }
//        std::cout << '\n';

        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform);
//
        glDrawElements(GL_LINES, line_indices.size() , GL_UNSIGNED_INT, 0);
//
//        scale = 0.3;


//
//        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform2);
//
//        glDrawElements(GL_TRIANGLES, std::size(cube_indices) , GL_UNSIGNED_INT, 0);
//
//        scale = 0.1;
//        float transform3[16] =
//                {
//                        cos(angle) * scale, 0.f, sin(angle) * scale, cube_x - 1.f,
//                        0.f, 1.f * scale, 0.f, cube_y - 1.f,
//                        -sin(angle) * scale, 0.f, cos(angle) * scale, -z_shift,
//                        0.f, 0.f, 0.f, 1.f,
//                };
//
//
//        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform3);
//
//        glDrawElements(GL_TRIANGLES, std::size(cube_indices) , GL_UNSIGNED_INT, 0);
//


        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}