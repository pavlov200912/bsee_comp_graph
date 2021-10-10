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

std::string to_string(std::string_view str) {
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message) {
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
        R"(#version 330 core
uniform mat4 view;
uniform mat4 transform;
layout (location = 0) in vec2 in_position_xz;
layout (location = 1) in vec3 in_color;
layout (location = 2) in float in_position_y;

out vec4 color;
void main()
{
	gl_Position = view * transform * vec4(in_position_xz[0], in_position_y, in_position_xz[1], 1.0);
	color = vec4(in_color, 1.0);
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

GLuint create_shader(GLenum type, const char *source) {
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

struct vec3 {
    float x;
    float y;
    float z;
};

struct rgb {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};


struct vec2 {
    float x;
    float y;
};

struct vertex {
    vec2 position_xz;
    float position_y;
    std::uint8_t color[4];
};


// Square matrix multiplication
void matrix_multiply(const float *a, const float *b, float *c, size_t size) {
    std::fill(c, c + size * size, 0);
    for (int i = 0; i < size; i++) {
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


// [0, 1] * [0, 1] -> [0, 1]

float metaballs(float x, float y, float time, const std::vector<vec3>& circles, const std::vector<float>& weights) {
    float x_point = 20.0f * float(x) - 10.0f; // [0, 1] -> [-3, 3]
    float y_point = 20.0f * float(y) - 10.0f;
    float result = 0.f;
    assert(circles.size() == weights.size());
    for (int i = 0; i < circles.size(); i++) {
        result += weights[i] * exp(- (pow(x_point - circles[i].x, 2) + pow(y_point - circles[i].y, 2)) / pow(circles[i].z * 3, 2));
    }
    return result / 5 + 0.5 + sin((x_point + time) / 4) / 10 + sin((y_point + time) / 5) / 10;
}

float function(float x, float y, float time) {
    float x_point = 6.0f * float(x) - 3.0f; // [0, 1] -> [-3, 3]
    float y_point = 6.0f * float(y) - 3.0f;
    return (-x_point * x_point - y_point * y_point + 18) / 18 + sinf(time);
}

float grid_coordinate_to_float(int x, int grid_size) {
    return (float) x / grid_size;
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
                                 grid_coordinate_to_float(z, grid_size),
                         },
                         grid_coordinate_to_float(y, grid_size),
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

float linear_interpolation(float x1, float y1, float x2, float y2, float y) {
    if (x1 > x2) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    if (y1 < y2) {
        // (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)
        // x = (y - y1) / (y2 - y1) * (x2 - x1) + x1
        float x = (y - y1) / (y2 - y1) * (x2 - x1) + x1;
        assert(x1 <= x and x <= x2);
        return x;
    } else {
        // (y - y2) / (y1 - y2) = (x2 - x) / (x2 - x1)
        // x = (y - y2) / (y1 - y2) * (x1 - x2) + x2
        float x = (y - y2) / (y1 - y2) * (x1 - x2) + x2;
        assert(x1 <= x and x <= x2);
        return x;
    }
}

struct pair_hash {
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2;
    }
};

enum Direction {
    TOP,
    LEFT,
    RIGHT,
    BOTTOM
};


std::pair<std::pair<int, int>, std::pair<int, int>> dir_to_ints(Direction d) {
    switch (d) {
        case TOP:
            return {{0, 0},
                    {0, 1}};
        case RIGHT:
            return {{0, 1},
                    {1, 1}};
        case LEFT:
            return {{0, 0},
                    {1, 0}};
        case BOTTOM:
            return {{1, 0},
                    {1, 1}};
    }
}

int push_edge(Direction d, int i, int j, float level, int graph_size,
              std::vector<vertex> &vertices, std::unordered_map<std::pair<int, int>, int, pair_hash> &edges_index,
              const std::vector<float> &function_values) {
    auto[edge1, edge2] = dir_to_ints(d);
    int i1 = i + edge1.first, j1 = j + edge1.second;
    int i2 = i + edge2.first, j2 = j + edge2.second;
    int index = 0;
    std::pair<int, int> edge = {i1 * graph_size + j1, i2 * graph_size + j2};
    if (edges_index.contains(edge)) {
        index = edges_index[edge];
    } else {
        vertex v1{};
        v1.color[0] = 0;
        v1.color[1] = 0;
        v1.color[2] = 0;
        v1.color[3] = 0;

        auto point_x1 = grid_coordinate_to_float(j1, graph_size);
        auto point_z1 = grid_coordinate_to_float(i1, graph_size);

        auto point_x2 = grid_coordinate_to_float(j2, graph_size);
        auto point_z2 = grid_coordinate_to_float(i2, graph_size);

        v1.position_xz = {
                linear_interpolation(point_x1, function_values[i1 * graph_size + j1], point_x2,
                                     function_values[i2 * graph_size + j2], level),
                linear_interpolation(point_z1, function_values[i1 * graph_size + j1], point_z2,
                                     function_values[i2 * graph_size + j2], level),
        };
        v1.position_y = level + 0.001f;

        edges_index[edge] = int(vertices.size());
        index = int(vertices.size());
        vertices.push_back(v1);
    }
    return index;
}

void build_isolines(int graph_size,
                    std::vector<vertex> &isolines_vertex,
                    std::vector<std::uint32_t> &isolines_index,
                    const std::vector<float> &function_values, const std::vector<float> &isolines_levels) {
    std::unordered_map<std::pair<int, int>, int, pair_hash> edges_index;
    std::vector<bool> is_higher(graph_size * graph_size, true);

    for (auto level: isolines_levels) {

        for (int i = 0; i < graph_size; i++) {
            for (int j = 0; j < graph_size; j++) {
                if (!is_higher[j + i * graph_size]) continue;
                auto point_z = function_values[i * graph_size + j];
                if (point_z < level)
                    is_higher[j + i * graph_size] = false;
            }
        }
        edges_index.clear();
        for (int i = 0; i < graph_size - 1; i++) {
            for (int j = 0; j < graph_size - 1; j++) {
                // 8 4
                // 1 2
                int marching_case = (is_higher[j + i * graph_size] << 3) | (is_higher[j + 1 + i * graph_size] << 2);
                marching_case |= (is_higher[j + (i + 1) * graph_size]) | (is_higher[j + 1 + (i + 1) * graph_size] << 1);

                // For each 1 0 edge
                // Check if there is an edges_index already
                // If it is, find that vertex
                // If it is not, create new vertex and push it, update edges_index
                // Find vertices for both 1 0

                switch (marching_case) {
                    case 0:
                        break;
                    case 1: {
                        int index1 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }
                    case 2: {
                        int index1 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }
                    case 3: {

                        int index1 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
//                        std::cout << "Connect: " << index1 << ' ' << index2 << '\n';
                        break;
                    }

                    case 4: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }

                    case 5: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);

                        int index3 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index4 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index3);
                        isolines_index.push_back(index4);

                        break;
                    }
                    case 6: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);

                        break;
                    }
                    case 7: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);

                        break;
                    }
                    case 8: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);

                        break;
                    }
                    case 9: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);

                        break;
                    }
                    case 10: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);

                        int index3 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index4 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index3);
                        isolines_index.push_back(index4);
                        break;
                    }
                    case 11: {
                        int index1 = push_edge(TOP, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }
                    case 12: {
                        int index1 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }
                    case 13: {
                        int index1 = push_edge(RIGHT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }
                    case 14: {
                        int index1 = push_edge(LEFT, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        int index2 = push_edge(BOTTOM, i, j, level, graph_size, isolines_vertex, edges_index,
                                               function_values);
                        isolines_index.push_back(index1);
                        isolines_index.push_back(index2);
                        break;
                    }
                    case 15: {
                        break;
                    }
                    default:
                        std::cout << "Something wrong...\n";
                }

            }
        }
    }

}

void build_grid(std::vector<vertex> &grid_vertices,
                std::vector<std::uint32_t> &line_indices,
                int grid_size) {

    std::unordered_map<int, int> grid_vertex_to_index;
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
        push_grid_vertex(0, i, 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(0, i, grid_size - 1, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(0, grid_size - 1, i, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);

        // Y plane
        push_grid_vertex(i, grid_size - 1, 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
        push_grid_vertex(grid_size - 1, i, 0, grid_size, grid_vertices, grid_vertex_to_index, grid_corners);
    }

    // MAIN GRID
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(grid_size - 1, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, grid_size - 1, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, 0, grid_size)]);
    line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, grid_size - 1, grid_size)]);


    for (int i = 0; i < grid_size; i++) {
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, 0, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, 0, grid_size - 1, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, 0, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(i, grid_size - 1, 0, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, i, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(grid_size - 1, i, 0, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, i, 0, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, i, grid_size - 1, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, i, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(grid_size - 1, 0, i, grid_size)]);

        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, 0, i, grid_size)]);
        line_indices.push_back(grid_vertex_to_index[grid_vertex_to_int(0, grid_size - 1, i, grid_size)]);
    }

}

float max(float a, float b) {
    return a > b ? a : b;
}


float min(float a, float b) {
    return a > b ? b : a;
}

int main() try {
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

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 4",
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
    std::vector<vertex> grid_vertices;
    std::vector<std::uint32_t> line_indices;

    build_grid(grid_vertices, line_indices, grid_size);

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


    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, line_indices.size() * sizeof(line_indices[0]), line_indices.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex),
                          reinterpret_cast<void *>(offsetof(vertex, position_xz)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex),
                          reinterpret_cast<void *>(offsetof(vertex, color)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex),
                          reinterpret_cast<void *>(offsetof(vertex, position_y)));





    std::vector<vec2> graph_vertices_xz;
    std::vector<rgb> graph_colors;

    int graph_size = 120;
//    std::vector<std::vector<float>> function_values(graph_size, std::vector<float>(graph_size, 0.f));
    std::vector<float> function_values;
    for (int i = 0; i < graph_size; i++) {
        for (int j = 0; j < graph_size; j++) {
            auto point_x = grid_coordinate_to_float(j, graph_size);
            auto point_y = grid_coordinate_to_float(i, graph_size);
            auto point_z = metaballs(point_x, point_y, 0, {{0, 0, 0.25}}, {1});
            function_values.push_back(point_z);
            graph_vertices_xz.push_back(
                    {
                            point_x,
                            point_y,
                    }

            );
            graph_colors.push_back(
                    {
                            std::uint8_t(255 * min(1.f, abs(1 - point_z))),
                            std::uint8_t(255 * min(1.f, abs(1 - point_z))),
                            255,
                    }
            );
        }
    }


    std::vector<std::uint32_t> graph_triangles;
    for (int i = 0; i < graph_size - 1; i++) {
        for (int j = 0; j < graph_size - 1; j++) {
            graph_triangles.push_back(i * graph_size + j);
            graph_triangles.push_back(i * graph_size + graph_size + j);
            graph_triangles.push_back(i * graph_size + j + 1);

            graph_triangles.push_back(i * graph_size + j + 1);
            graph_triangles.push_back(i * graph_size + graph_size + j);
            graph_triangles.push_back(i * graph_size + graph_size + j + 1);
        }
    }


    GLuint vbo_xz;
    glGenBuffers(1, &vbo_xz);

    GLuint vbo_y;
    glGenBuffers(1, &vbo_y);

    GLuint vbo_rgba;
    glGenBuffers(1, &vbo_rgba);


    GLuint graph_vao;
    glGenVertexArrays(1, &graph_vao);
//    glBindBuffer(GL_ARRAY_BUFFER, vbo_xz);
    glBindVertexArray(graph_vao);




    glBindBuffer(GL_ARRAY_BUFFER, vbo_xz);

    glBufferData(GL_ARRAY_BUFFER,
                 graph_vertices_xz.size() * sizeof(graph_vertices_xz[0]),
                 graph_vertices_xz.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2),
                          reinterpret_cast<void *>(0));


    glBindBuffer(GL_ARRAY_BUFFER, vbo_y);

    glBufferData(GL_ARRAY_BUFFER,
                 function_values.size() * sizeof(function_values[0]),
                 function_values.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float),
                          reinterpret_cast<void *>(0));


    glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba);

    glBufferData(GL_ARRAY_BUFFER,
                 graph_colors.size() * sizeof(graph_colors[0]),
                 graph_colors.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(rgb), reinterpret_cast<void*>(0));



    GLuint graph_ebo;
    glGenBuffers(1, &graph_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graph_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, graph_triangles.size() * sizeof(graph_triangles[0]), graph_triangles.data(),
                 GL_STATIC_DRAW);

    GLuint isolines_vbo;
    glGenBuffers(1, &isolines_vbo);

    std::vector<vertex> isolines_vertex;
    std::vector<std::uint32_t> isolines_index;
    std::vector<float> isolines_levels; // should be sorted from lower to higher

    int isolines_size = 20;
    for (int i = 1; i <= isolines_size; i++) {
        isolines_levels.push_back(float(i) / float(isolines_size));
    }


    build_isolines(graph_size, isolines_vertex, isolines_index, function_values, isolines_levels);


    glBindBuffer(GL_ARRAY_BUFFER, isolines_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 isolines_vertex.size() * sizeof(isolines_vertex[0]),
                 isolines_vertex.data(), GL_DYNAMIC_DRAW);

    GLuint isolines_vao;
    glGenVertexArrays(1, &isolines_vao);
    glBindBuffer(GL_ARRAY_BUFFER, isolines_vbo);
    glBindVertexArray(isolines_vao);


    GLuint isolines_ebo;
    glGenBuffers(1, &isolines_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, isolines_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, isolines_index.size() * sizeof(isolines_index[0]), isolines_index.data(),
                 GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex),
                          reinterpret_cast<void *>(offsetof(vertex, position_xz)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex),
                          reinterpret_cast<void *>(offsetof(vertex, position_y)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex),
                          reinterpret_cast<void *>(offsetof(vertex, color)));



    std::vector<rgb> color_set;
    color_set.push_back({255, 255, 255});
    for (int i = 0; i < 10; i++) {
        color_set.push_back({255, std::uint8_t(255 - 25 * i), 255});
    }
    color_set.push_back({255, 0, 255});
    for (int i = 0; i < 10; i++) {
        color_set.push_back({255, 0, std::uint8_t(255 - 25 * i)});
    }
    color_set.push_back({255, 0, 0});
    for (int i = 0; i < 10; i++) {
        color_set.push_back({255, std::uint8_t(25 * i), 0});
    }
    color_set.push_back({255, 255, 0});
    for (int i = 0; i < 10; i++) {
        color_set.push_back({255, 255, std::uint8_t(25 * i)});
    }
    color_set.push_back({255, 255, 255});


    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    glEnable(GL_DEPTH_TEST);

    bool running = true;

    float cube_x = 0;
    float cube_y = 0;
    float dt = 0.f;
    float speed = 2.f;
    float z_shift = 1.5;

    float y_angle = 5.45;
    float x_angle = -0.5;

    float scale = 0.5f;

    int iteration = 0;

    bool update_graph = false;
    bool update_isolines = false;

    while (running) {
        iteration += 1;
        glClear(GL_DEPTH_BUFFER_BIT);

        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT:
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            glViewport(0, 0, width, height);
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_LEFT) {
                        cube_x -= speed * dt;
                    } else if (event.key.keysym.sym == SDLK_RIGHT) {
                        cube_x += speed * dt;
                    } else if (event.key.keysym.sym == SDLK_UP) {
                        cube_y += speed * dt;
                    } else if (event.key.keysym.sym == SDLK_DOWN) {
                        cube_y -= speed * dt;
                    } else if (event.key.keysym.sym == SDLK_s) {
//                        z_shift += speed * dt;
                        x_angle += speed * dt;
                    } else if (event.key.keysym.sym == SDLK_w) {
//                        z_shift -= speed * dt;
                        x_angle -= speed * dt;
                    } else if (event.key.keysym.sym == SDLK_d) {
                        y_angle -= speed * dt;
                    } else if (event.key.keysym.sym == SDLK_a) {
                        y_angle += speed * dt;
                    } else if (event.key.keysym.sym == SDLK_SPACE) {
                        scale += speed * dt;
                    } else if (event.key.keysym.sym == SDLK_LSHIFT) {
                        scale -= speed * dt;
                    } else if (event.key.keysym.sym == SDLK_MINUS) {
                        std::cout << "Graph size changed from " << graph_size << " to ";
                        graph_size = int(max(1.0, graph_size - 1));
                        std::cout << graph_size << '\n';
                        update_graph = true;
                    } else if (event.key.keysym.sym == SDLK_EQUALS) {
                        std::cout << "Graph size changed from " << graph_size << " to ";
                        graph_size += 1;
                        std::cout << graph_size << '\n';
                        update_graph = true;
                    } else if (event.key.keysym.sym == SDLK_1) {
                        std::cout << "Isoline size changed from " << isolines_size << " to ";
                        isolines_size = int(max(1, isolines_size - 1));
                        std::cout << isolines_size << '\n';
                        update_isolines = true;
                    } else if (event.key.keysym.sym == SDLK_2) {
                        std::cout << "Isoline size changed from " << isolines_size << " to ";
                        isolines_size += 1;
                        std::cout << isolines_size << '\n';
                        update_isolines = true;
                    }
                    break;
                case SDL_KEYUP:
                    button_down[event.key.keysym.sym] = false;
                    break;
            }

        if (!running)
            break;

        if (update_graph) {
            graph_vertices_xz.clear();
            for (int i = 0; i < graph_size; i++) {
                for (int j = 0; j < graph_size; j++) {
                    auto point_x = grid_coordinate_to_float(j, graph_size);
                    auto point_y = grid_coordinate_to_float(i, graph_size);
                    graph_vertices_xz.push_back(
                            {
                                    point_x,
                                    point_y,
                            }

                    );
                }
            }

            glBindBuffer(GL_ARRAY_BUFFER, vbo_xz);

            glBufferData(GL_ARRAY_BUFFER,
                         graph_vertices_xz.size() * sizeof(graph_vertices_xz[0]),
                         graph_vertices_xz.data(), GL_STATIC_DRAW);

            graph_triangles.clear();
            for (int i = 0; i < graph_size - 1; i++) {
                for (int j = 0; j < graph_size - 1; j++) {
                    graph_triangles.push_back(i * graph_size + j);
                    graph_triangles.push_back(i * graph_size + graph_size + j);
                    graph_triangles.push_back(i * graph_size + j + 1);

                    graph_triangles.push_back(i * graph_size + j + 1);
                    graph_triangles.push_back(i * graph_size + graph_size + j);
                    graph_triangles.push_back(i * graph_size + graph_size + j + 1);
                }
            }

            std::cout << "Graph updated\n";

            update_graph = false;
        }

        if (update_isolines) {
            std::cout << "Isolines updated\n";
            isolines_levels.clear();
            for (int i = 1; i <= isolines_size; i++) {
                isolines_levels.push_back(float(i) / float(isolines_size));
            }
            update_isolines = false;
        }

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

        float view[16] =
                {
                        2 * near / (right - left), 0.f, (right + left) / (right - left), 0.f,
                        0.f, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0.f,
                        0.f, 0.f, -(far + near) / (far - near), -(2 * far * near) / (far - near),
                        0.f, 0.f, -1.f, 0.f,
                };

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
        matrix_multiply(shift_matrix, scale_matrix, scale_shift, 4);
        matrix_multiply(x_rotate, y_rotate, rotate, 4);
        matrix_multiply(scale_shift, rotate, transform, 4);

        //         Update function

        std::vector<vec3> circles = {{3 * sinf(time), 3 * cosf(time) * sinf(time), 1},
                                     {3 * cosf(time), 3 * cosf(time) * sinf(time), 1},
                                     {3 * sinf(time), 3 * cosf(time), 0.5}};
        std::vector<float> weigths = {1, -1, -2};
        {
            function_values.clear();
            for (int i = 0; i < graph_size; i++) {
                for (int j = 0; j < graph_size; j++) {
                    auto point_x = grid_coordinate_to_float(j, graph_size);
                    auto point_y = grid_coordinate_to_float(i, graph_size);
                    auto point_z = metaballs(point_x, point_y, time, circles, weigths);
                    function_values.push_back(point_z);
                }
            }


        }

        // update isolines
        {
            isolines_vertex.clear();
            isolines_index.clear();
            build_isolines(graph_size, isolines_vertex, isolines_index, function_values, isolines_levels);
        }


        // update colors
        {
            graph_colors.clear();
            for (int i = 0; i < graph_size; i++) {
                for (int j = 0; j < graph_size; j++) {
                    auto point_x = grid_coordinate_to_float(j, graph_size);
                    auto point_y = grid_coordinate_to_float(i, graph_size);
                    auto point_z = metaballs(point_x, point_y, time, circles, weigths);
                    function_values.push_back(point_z);

                    int i1 = int(point_z * color_set.size()) % color_set.size();
                    int i2 = int(max(i1 - 1, 0)) % color_set.size();
                    rgb c1 = color_set[i1];
                    rgb c2 = color_set[i2];
                    rgb new_color = {
                            std::uint8_t((c1.r + c2.r) / 2),
                            std::uint8_t((c1.g + c2.g) / 2),
                            std::uint8_t((c1.b + c2.b) / 2)
                    };
                    graph_colors.push_back(new_color);

                }
            }

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graph_ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, graph_triangles.size() * sizeof(graph_triangles[0]), graph_triangles.data(),
                         GL_STATIC_DRAW);

        }

        // load colors
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_rgba);

            glBufferData(GL_ARRAY_BUFFER,
                         graph_colors.size() * sizeof(graph_colors[0]),
                         graph_colors.data(), GL_STATIC_DRAW);

        }

        // load function
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_y);
            glBufferData(
                    GL_ARRAY_BUFFER,
                    sizeof(function_values[0]) * function_values.size(),
                    function_values.data(),
                    GL_DYNAMIC_DRAW
            );
        }

        // load isolines
        {
            glBindBuffer(GL_ARRAY_BUFFER, isolines_vbo);
            glBufferData(GL_ARRAY_BUFFER,
                         isolines_vertex.size() * sizeof(isolines_vertex[0]),
                         isolines_vertex.data(), GL_DYNAMIC_DRAW);


            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, isolines_ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, isolines_index.size() * sizeof(isolines_index[0]),
                         isolines_index.data(),
                         GL_DYNAMIC_DRAW);
        }

        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform);


        glBindVertexArray(vao);
        glDrawElements(GL_LINES, line_indices.size(), GL_UNSIGNED_INT, 0);


        glBindVertexArray(graph_vao);
        glDrawElements(GL_TRIANGLES, graph_triangles.size(), GL_UNSIGNED_INT, 0);


        glBindVertexArray(isolines_vao);
        glDrawElements(GL_LINES, isolines_index.size(), GL_UNSIGNED_INT, 0);


        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}