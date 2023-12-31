#include <vulkan/vulkan.h> // LunerG sdk header - off-screen rendering
#define GLFW3_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <stdexcept> // for reporting errors
#include <cstdlib>

#define ENABLE_VALIDATION_LAYER 1

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow(); // init glfw window
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow() {
        glfwInit(); // initialize first.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // disable OpenGL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // disable resize screen

        window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "VKRenderer", nullptr, /*related to opengl*/nullptr);

    }

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // set app info enum variant to struct Type field
        appInfo.pApplicationName = "VKRenderer"; // application name
        appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0); // version specifier
        appInfo.pEngineName = "NO ENGINE";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3; // VKRenderer using sdk version 1.3.268

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount); // return Vulkan Instance extensions required by GLFW , type is array of const char*


       /* DEBUG : print glfwExtensions
       for (int i = 0; i < glfwExtensionCount; i++) {
            std::cout << glfwExtensions[i] << std::endl;
        }
        */

        // TODO : what is exact need for this struct?
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        createInfo.enabledLayerCount = 0; // global validation layer

        // At this point, we specify everything we need for an instance.
        if (vkCreateInstance(&createInfo, /*custom allocator*/ nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create an instance");
        }
    }

    // DEBUG : print all extension properties
    void debugExtensionProperties() {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr); // enumerate available extensions and store the count into 'extensionCount'
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()); // fill extension data into vector
        for (auto& name : extensions) {
            // print extension names.
            std::cout << name.extensionName << " is  available" << std::endl;
        }
    }

    bool checkValidationLayerSupport() {
#ifdef ENABLE_VALIDATION_LAYER
        return true;
#endif // ENABLE_VALIDATION_LAYER
        return false;
    }

    void initVulkan() {
        // init private Vulkan objects here
        createInstance();
    }

    void mainLoop() {
        // rendering frames

        // keep window while application running
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);

        glfwTerminate();
    }

private:
    GLFWwindow* window;
    VkInstance instance;
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        // propagate errors back to main thread
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE; // cstdlib macro for failure
    }

    return EXIT_SUCCESS; // cstdlib macro for success
}