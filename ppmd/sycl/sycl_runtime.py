import ctypes
from ppmd import abort
from ppmd.sycl.sycl_build import sycl_simple_lib_creator, build_static_libs

# try:
#     LIB_HELPER = build_static_libs('syclHelperLib')
# except Exception as e:
#     print("LIB_HELPER load error", e)
#     abort('sycl_runtime error: Module is not initialised correctly, '
#           'SYCL helper lib not loaded')

LIB_HELPER = build_static_libs('syclHelperLib')

class Device:
    def __init__(self, device_selector='cpu_selector'):
        
        h = """
        """
        s = r"""
        using namespace sycl;
        
        device d;

        extern "C"
        int device_selector(device **d_ptr){{
            try {{
                d = device({SELECTOR}());
            }} catch (exception const &e) {{
                std::cout << e.what() << "\n";
            }}

            *d_ptr = &d;
            return 0;
        }}
        """.format(
            SELECTOR=device_selector
        )

        lib = sycl_simple_lib_creator(h, s, "device_selector")['device_selector']
        
        self.device = ctypes.c_void_p()
        lib(ctypes.byref(self.device))

class Queue:
    def __init__(self, device):
        self.device = device
        self.lib = sycl_simple_lib_creator(
                "",
                r"""
                sycl::queue q;
                extern "C"
                int queue_creator(sycl::device *d, sycl::queue **q_ptr){
                    q = sycl::queue(*d);
                    *q_ptr = &q;
                    return 0;
                }
                extern "C"
                int print_device_name(sycl::queue *q_ptr){
                    sycl::queue q_tmp = *q_ptr;
                    std::cout << q_tmp.get_device().get_info<sycl::info::device::name>() << "\n";
                    return 0;
                }
                """,
                "queue_creator"
            )
        
        self.queue = ctypes.c_void_p()
        self.lib["queue_creator"](self.device.device, ctypes.byref(self.queue))

    def print_device_name(self):
        self.lib["print_device_name"](self.queue)

device = Device()
queue = Queue(device)
queue.print_device_name()








