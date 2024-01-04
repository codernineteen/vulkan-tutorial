#include <vulkan/vulkan.h> // LunerG sdk header - off-screen rendering
#define GLFW3_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <optional>
#include <set>
#include <vector>
#include <stdexcept> // for reporting errors
#include <cstdlib>
#include <cstring> 

const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG // if not debug mode
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

// a proxy function to look up adress of vkCreateDebugUtilsMessengerEXT
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger
) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// deallocator for VkDebugUtilsMessengerEXT object
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

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
    //----------------------------Instance and Extensions---------------------------------
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // setup callback by using debug extension
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( // VKAPI_ATTR and VKAPI_CALL ensure correct function signature
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, // severity of message. we can use comparision operator to check severity.
        VkDebugUtilsMessageTypeFlagsEXT messageType, // message types : unrelated to spec , violate spec , potential non-optimal use of api
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, // member of this struct is 'pMessage, pObjects, objectCount'
        void* pUserData // custom data passed from vkCreateDebugUtilsMessengerEXT
    ) {

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    void createInstance() {
        // if validation layers are enabled but there is no validation layer support, throw error
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available");
        }


        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // set app info enum variant to struct Type field
        appInfo.pApplicationName = "VKRenderer"; // application name
        appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0); // version specifier
        appInfo.pEngineName = "NO ENGINE";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3; // VKRenderer using sdk version 1.3.268

        auto extensions = getRequiredExtensions();
        // TODO : what is exact need for this struct?
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = extensions.size();
        createInfo.ppEnabledExtensionNames = extensions.data();

        // By creating additional debug messenger,it will automatically be used during vkCreateInstance and vkDestroyInstance and cleaned up after that.
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }


        // At this point, we specify everything we need for an instance.
        if (vkCreateInstance(&createInfo, /*custom allocator*/ nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create an instance");
        }
    }

    //----------------------------Validation Layer---------------------------------
    
    // check if all the requested layers are available
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()); // get all available layers

        for (const char* layerName : validationLayers) {
			bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) { // if layer name is same as requested
                	layerFound = true;
                	break;
                }
			}

            if (!layerFound) {
				return false;
			}
		}

        return true;
    }
    
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if(!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger");
		}
    }

    //----------------------------Physical/Logical Device---------------------------------
    
    // Need to check if device is suitable for rendering
    bool isDeviceSuitable(VkPhysicalDevice device) {
        // just use any GPU for now. can implement suitability checker in a various way later.
        QueueFamilyIndices indices = findQueueFamilies(device);
        return indices.isComplete();
	}

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr); // query numver of devices
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount); // allocate vector with deviceCount size
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()); // get all devices

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
				physicalDevice = device; // if we find a suitable device, break the loop
				break;
			}
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU");
        }
    }

    struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily; // for presentation

        bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0; // query number of queue families
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data()); // get all queue families

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) { // check if queue family supports graphics
    			indices.graphicsFamily = i;
        	}
            if (presentSupport) {
                indices.graphicsFamily = i;
            }
            // early exit if we find a queue family that supports graphics
            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // TODO : specify the set of device features that we'll be using
        VkPhysicalDeviceFeatures deviceFeatures{}; // struct to specify features we want to use

		// create logical device (naub structure)
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data(); // pointer to queue create info
        createInfo.queueCreateInfoCount = queueCreateInfos.size();
        createInfo.pEnabledFeatures = &deviceFeatures; // pointer to device features
        createInfo.enabledExtensionCount = 0;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(/*physical device to interface with*/physicalDevice, &createInfo, nullptr, /*pointer to store logical device handle*/&device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device");
		}

        // retrieve queue handles
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue); // get graphics queue
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue); // get presentation queue
    }

    //----------------------------Surface---------------------------------
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, /*pointer go VkSurfaceKHR*/&surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void initVulkan() {
        // init private Vulkan objects here
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void mainLoop() {
        // rendering frames

        // keep window while application running
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        vkDestroySurfaceKHR(instance, surface, nullptr); // destroy surface before destroying instance
        vkDestroyInstance(instance, nullptr);
        vkDestroyDevice(device, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

private:
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger; // callback is a part of debug messenger
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // handle to physical device
    VkDevice device; // logical device
    VkQueue graphicsQueue; // a handle to the graphics queue, implicitly destroyed when logical device is destroyed
    VkQueue presentQueue; // a handle to the presentation queue
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