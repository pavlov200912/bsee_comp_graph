#ifdef WIN32
#include <SDL.h>
#undef main
#else

#include <SDL2/SDL.h>

#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
#include "shader.h"
#include "mesh.h"
#include "model.h"


#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

std::string to_string(std::string_view str) {
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message) {
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader[] = R"(
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 in_texcoord;

out vec2 tex_coords;
out vec3 position;
out vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    tex_coords = in_texcoord;
	position = (model * vec4(in_position, 1.0)).xyz;
    normal = a_normal;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
)";

const char fragment_shader[] = R"(
#version 330 core
out vec4 out_color;
uniform vec3 ambient;
uniform vec3 light_position[5];
uniform vec3 light_color[5];
uniform vec3 light_attenuation[5];
uniform mat4 transform;
uniform sampler2D texture_diffuse1;
uniform sampler2D shadow_map;

in vec3 position;
in vec2 tex_coords;
in vec3 normal;

vec3 light_source(vec3 position1, vec3 light_position1, vec3 light_color1, vec3 light_attenuation1) {
    vec3 light_vector = light_position1 - position1;
    vec3 light_direction = normalize(light_vector);
    float cosine = dot(normal, light_direction);
    float light_factor = max(0.0, cosine);

    float light_distance = length(light_vector);
    float light_intensity = 1.0 / dot(light_attenuation1, vec3(1.0, light_distance, light_distance * light_distance));
    return light_factor * light_intensity * light_color1;
}

void main()
{
    vec4 shadow_pos = transform * vec4(position, 1.0);
	shadow_pos /= shadow_pos.w;
	shadow_pos = shadow_pos * 0.5 + vec4(0.5);

    bool in_shadow_texture = (shadow_pos.x > 0.0) && (shadow_pos.x < 1.0) && (shadow_pos.y > 0.0) && (shadow_pos.y < 1.0) && (shadow_pos.z > 0.0) && (shadow_pos.z < 1.0);
	float shadow_factor = 1.0;

	if (in_shadow_texture)
		shadow_factor = (texture(shadow_map, shadow_pos.xy).r < shadow_pos.z - 0.005) ? 0.0 : 1.0;

    vec3 result_color = vec3(0.0, 0.0, 0.0);
    result_color += ambient;
    result_color += light_source(position, light_position[0], light_color[0], light_attenuation[0]) * shadow_factor;
    for (int i = 1; i < 5; i++) {
        result_color += light_source(position, light_position[i], light_color[i], light_attenuation[i]);
    }
    result_color = texture(texture_diffuse1, tex_coords).rgb * result_color;
    out_color = vec4(result_color, 1.0);
}
)";

const char normal_vertex_shader[] = R"(
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 in_texcoord;

out vec2 tex_coords;
out vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    tex_coords = in_texcoord;
	position = (model * vec4(in_position, 1.0)).xyz;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
)";

const char normal_fragment_shader[] = R"(
#version 330 core
out vec4 out_color;
uniform vec3 ambient;
uniform mat4 model;
uniform mat4 transform;
uniform vec3 light_position[5];
uniform vec3 light_color[5];
uniform vec3 light_attenuation[5];
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_normal1;
uniform sampler2D shadow_map;


in vec3 position;
in vec2 tex_coords;

vec3 normal;

vec3 light_source(vec3 position1, vec3 light_position1, vec3 light_color1, vec3 light_attenuation1) {
    vec3 light_vector = light_position1 - position1;
    vec3 light_direction = normalize(light_vector);
    float cosine = dot(normal, light_direction);
    float light_factor = max(0.0, cosine);

    float light_distance = length(light_vector);
    float light_intensity = 1.0 / dot(light_attenuation1, vec3(1.0, light_distance, light_distance * light_distance));
    return light_factor * light_intensity * light_color1;
}

void main()
{
    normal = (vec4(2 * texture(texture_normal1, tex_coords).rgb - 1.0, 0.0)).xyz;

    vec4 shadow_pos = transform * vec4(position, 1.0);
	shadow_pos /= shadow_pos.w;
	shadow_pos = shadow_pos * 0.5 + vec4(0.5);

    bool in_shadow_texture = (shadow_pos.x > 0.0) && (shadow_pos.x < 1.0) && (shadow_pos.y > 0.0) && (shadow_pos.y < 1.0) && (shadow_pos.z > 0.0) && (shadow_pos.z < 1.0);
	float shadow_factor = 1.0;
	if (in_shadow_texture)
		shadow_factor = (texture(shadow_map, shadow_pos.xy).r < shadow_pos.z - 0.005) ? 0.0 : 1.0;


    vec3 result_color = vec3(0.0, 0.0, 0.0);
    result_color += ambient;
    result_color += light_source(position, light_position[0], light_color[0], light_attenuation[0]) * shadow_factor;
    for (int i = 1; i < 5; i++) {
        result_color += light_source(position, light_position[i], light_color[i], light_attenuation[i]);
    }
    result_color = texture(texture_diffuse1, tex_coords).rgb * result_color;
    out_color = vec4(result_color, 1.0);
}
)";

const char shadow_vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;

void main()
{
	gl_Position = transform * model * vec4(in_position, 1.0);
}
)";

const char shadow_fragment_shader_source[] =
        R"(#version 330 core

void main()
{}
)";


const char debug_vertex_shader_source[] =
        R"(#version 330 core

vec2 vertices[6] = vec2[6](
	vec2(-1.0, -1.0),
	vec2( 1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0,  1.0)
);

out vec2 texcoord;

void main()
{
	vec2 position = vertices[gl_VertexID];
	gl_Position = vec4(position * 0.25 + vec2(-0.75, -0.75), 0.0, 1.0);
	texcoord = position * 0.5 + vec2(0.5);
}
)";

const char debug_fragment_shader_source[] =
        R"(#version 330 core

uniform sampler2D shadow_map;

in vec2 texcoord;

layout (location = 0) out vec4 out_color;

void main()
{
	out_color = texture(shadow_map, texcoord);
}
)";

struct vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

int main() try {
    // Setting up window
    int width, height;
    SDL_Window *window;
    SDL_GLContext gl_context;
    {
        if (SDL_Init(SDL_INIT_VIDEO) != 0)
            sdl2_fail("SDL_Init: ");

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        window = SDL_CreateWindow("Graphics course practice 7",
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  800, 600,
                                  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

        if (!window)
            sdl2_fail("SDL_CreateWindow: ");

        SDL_GetWindowSize(window, &width, &height);

        gl_context = SDL_GL_CreateContext(window);
        if (!gl_context)
            sdl2_fail("SDL_GL_CreateContext: ");

        if (auto result = glewInit(); result != GLEW_NO_ERROR)
            glew_fail("glewInit: ", result);

        if (!GLEW_VERSION_3_3)
            throw std::runtime_error("OpenGL 3.3 is not supported");
    }

    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    ShaderProgram ourShader(vertex_shader, fragment_shader);
    ShaderProgram normalShader(normal_vertex_shader, normal_fragment_shader);

    // load models
    Model ourModel("/Users/Ivan.Pavlov/CLionProjects/dir1/graphics-course-practice/practice8/sponza1/sponza.obj");


    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;



    // Cam Position
    glm::vec3 cameraPos = glm::vec3(0.0f, 1.f, 0.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float yaw = -90.f;
    float pitch = 0.f;

    // Light
    glm::vec3 ambient = glm::vec3(0.5f, 0.5f, 0.5f);
    const int num_of_light = 5;
    glm::vec3 light_position[num_of_light] = {
            glm::vec3(0.0f, 5.0f, 0.0f), // SUN
            glm::vec3(11.19f, 2.38f, -4.27f),
            glm::vec3(-11.73f, 2.38f, -4.69f),
            glm::vec3(-12.23f, 2.38f, 4.61f),
            glm::vec3(11.25f, 2.38f, 4.08f),
    };
    glm::vec3 light_color[num_of_light] = {
            glm::vec3(5.0f, 5.0f, 5.0f), // SUN
            glm::vec3(0.1, 0.1, 0.5f),
            glm::vec3(0.5f, 0.f, 0.2f),
            glm::vec3(0.0f, 0.5f, 0.2f),
            glm::vec3(0.1f, 0.5f, 0.5f),
    };
    glm::vec3 light_attenuation[num_of_light] = {
            glm::vec3(1.0f, 0.00001, 0.01f), // SUN
            glm::vec3(1.0f, 0, 0.01f),
            glm::vec3(1.0f, 0, 0.01f),
            glm::vec3(1.0f, 0, 0.01f),
            glm::vec3(1.0f, 0, 0.01f),
    };

    // Shadows
    GLsizei shadow_map_resolution = 4096;

    GLuint shadow_map;
    glGenTextures(1, &shadow_map);
    glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_map_resolution, shadow_map_resolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    GLuint shadow_fbo;
    glGenFramebuffers(1, &shadow_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadow_map, 0);
    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Incomplete framebuffer!");
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    ShaderProgram shadow_program = ShaderProgram(shadow_vertex_shader_source, shadow_fragment_shader_source);

    for (int i = 0; i < ourModel.meshes.size(); i++) {
        ourModel.meshes[i].textures.push_back(
                Texture{
                    .id=shadow_map,
                    .type="shadow_map",
                    .path="shadow_map"
                }
                );
    }

    // Debug shadow

    ShaderProgram debug_program = ShaderProgram(debug_vertex_shader_source, debug_fragment_shader_source);
    GLuint debug_vao;
    glGenVertexArrays(1, &debug_vao);

    // Split meshes with and without normal maps
    std::vector<int> normal_meshes;
    std::vector<int> no_normal_meshes;
    for (int i = 0; i < ourModel.meshes.size(); i++) {
        auto mesh = ourModel.meshes[i];
        bool normal = false;
        for (int  j = 0; j < mesh.textures.size(); j++) {
            if (mesh.textures[j].type == "texture_normal") {
                normal = true;
                break;
            }
        }
        if (normal) {
            normal_meshes.push_back(i);
        } else {
            no_normal_meshes.push_back(i);
            std::cout << i << '\n';
        }
    }

    bool running = true;
    while (running) {
        // Keyboard events
        {
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
                        button_down[event.key.keysym.sym] = true;
                        break;
                    case SDL_KEYUP:
                        button_down[event.key.keysym.sym] = false;
                        break;
                }
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast < std::chrono::duration < float >> (now - last_frame_start).count();
        last_frame_start = now;
        time += dt;


        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(direction);

        // Keyboard response
        {
            float rotation_speed = 75.f;
            if (button_down[SDLK_UP])
                pitch += rotation_speed * dt;
            if (button_down[SDLK_DOWN])
                pitch -= rotation_speed * dt;

            if (button_down[SDLK_LEFT])
                yaw -= rotation_speed * dt;
            if (button_down[SDLK_RIGHT])
                yaw += rotation_speed * dt;

            float cameraSpeed = 0.1f;
            if (button_down[SDLK_w])
                cameraPos += cameraSpeed * cameraFront;

            if (button_down[SDLK_s])
                cameraPos -= cameraSpeed * cameraFront;

            if (button_down[SDLK_d])
                cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

            if (button_down[SDLK_a])
                cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        }

        glClearColor(0.8f, 0.8f, 0.9f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//        glm::vec3 light_direction = glm::normalize(glm::vec3(2.64, 2.43, 3.08)); //glm::normalize(glm::vec3(std::cos(time * 0.5f), 1.f, std::sin(time * 0.5f)));
        glm::vec3 light_direction = glm::normalize(glm::vec3(-11.5685, 27.2606, 2.32783));
        light_position[0] = glm::vec3(-11.5685, 27.2606, 2.32783);

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model = glm::mat4(1.f);
        float sponza_scale = 0.01f;
        glm::vec3 scale = glm::vec3(sponza_scale, sponza_scale, sponza_scale);
        model = glm::scale(model, scale);


        glm::mat4 view(1.f);

        view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 light_z = -light_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, {0.f, 1.f, 0.f}));
        glm::vec3 light_y = glm::cross(light_x, light_z);
        float shadow_scale = 0.03f;

        glm::mat4 transform = glm::mat4(1.f);
        for (size_t i = 0; i < 3; ++i)
        {
            transform[i][0] = shadow_scale * light_x[i];
            transform[i][1] = shadow_scale * light_y[i];
            transform[i][2] = shadow_scale * light_z[i];
        }



//        std::cout << cameraPos.x << ' ' << cameraPos.y << ' ' << cameraPos.z << '\n';

//         Draw shadows to shadow_map
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, shadow_map_resolution, shadow_map_resolution);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        shadow_program.use();

        shadow_program.setMatrix("model", model);
        shadow_program.setMatrix("transform", transform);

//        glBindVertexArray(vao);
//        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
        for (int i = 0; i < ourModel.meshes.size(); i++) {
            ourModel.meshes[i].Draw(shadow_program);
        }

        glBindTexture(GL_TEXTURE_2D, shadow_map);
        glGenerateMipmap(GL_TEXTURE_2D);


//        // Draw model
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
//
        glClearColor(0.8f, 0.8f, 0.9f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        ourShader.use();
        ourShader.setMatrix("projection", projection);
        ourShader.setMatrix("view", view);
        ourShader.setMatrix("model", model);
        ourShader.setMatrix("transform", transform);
        ourShader.setVec3("ambient", ambient);
        for (int i = 0; i < num_of_light; i++) {
            ourShader.setVec3("light_position[" + std::to_string(i) + "]", light_position[i]);
            ourShader.setVec3("light_color[" + std::to_string(i) + "]", light_color[i]);
            ourShader.setVec3("light_attenuation[" + std::to_string(i) + "]", light_attenuation[i]);
        }
//
        for (int mesh_id: no_normal_meshes) {
            ourModel.meshes[mesh_id].Draw(ourShader);
        }

        normalShader.use();
        normalShader.setMatrix("projection", projection);
        normalShader.setMatrix("view", view);
        normalShader.setMatrix("model", model);
        normalShader.setVec3("ambient", ambient);
        for (int i = 0; i < num_of_light; i++) {
            normalShader.setVec3("light_position[" + std::to_string(i) + "]", light_position[i]);
            normalShader.setVec3("light_color[" + std::to_string(i) + "]", light_color[i]);
            normalShader.setVec3("light_attenuation[" + std::to_string(i) + "]", light_attenuation[i]);
        }

        for (int mesh_id: normal_meshes) {
            ourModel.meshes[mesh_id].Draw(normalShader);
        }



        debug_program.use();
        debug_program.setInt("shadow_map", 0);
;
		glBindTexture(GL_TEXTURE_2D, shadow_map);
		glBindVertexArray(debug_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}