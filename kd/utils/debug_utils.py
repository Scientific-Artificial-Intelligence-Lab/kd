import os
import functools
import atexit

# Default no-op decorator that does nothing.
def _noop_decorator(func):
    return func

# The main 'profile' decorator, which will be replaced by a real profiler if enabled.
profile = _noop_decorator


# Default no-op function for print_obj_profile.
def print_obj_profile():
    pass


if os.getenv("MEM_DEBUG") == "1":
    try:
        from memory_profiler import profile as _mem_profile
        profile = _mem_profile
        print("✅ MEMORY_DEBUG enabled: using memory_profiler.")
    except ImportError:
        print("⚠️ MEMORY_DEBUG=1 but memory_profiler is not installed. Memory analysis disabled.")

if os.getenv("OBJ_DEBUG") == "1":
    try:
        import objgraph

        def print_obj_profile():
            """
            Manually callable function to print objgraph analysis.
            Use this in your code at specific points for fine-grained debugging.
            """
            print("\n" + "="*20 + " Objgraph Analysis " + "="*20)
            print("--- Most Common Types ---")
            objgraph.show_most_common_types(limit=15)
            print("\n" + "--- Object Growth Since Last Check ---")
            objgraph.show_growth(limit=10)

        # Optional: Still register for exit analysis as a fallback
        @atexit.register
        def _exit_obj_profile():
            print("\n" + "="*20 + " Final Objgraph Analysis at Exit " + "="*20)
            print_obj_profile()  # Reuse the function for consistency

        print("✅ OBJ_DEBUG enabled: using objgraph (call print_obj_profile() manually or check at exit).")

        # Examples:
        # Use case 1: Memory profiling
        #   Set MEMORY_DEBUG=1 in environment.
        #   In your script: from this_module import profile
        #   @profile
        #   def my_function(): ...
        #   Run the script: You'll see line-by-line memory usage printed.

        # Use case 2: Object graph analysis
        #   Set OBJ_DEBUG=1 in environment.
        #   In your script: from this_module import print_obj_profile
        #   def my_function():
        #       # Some code...
        #       print_obj_profile()  # Call here to analyze at this point
        #       # More code...
        #   Run the script: Objgraph info will print at the call point and at exit.

    except ImportError:
        print("⚠️ OBJ_DEBUG=1 but objgraph is not installed. Object analysis disabled.")