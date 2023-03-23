def attemptRegister(func1,func2):
    import torchvision.models
    try:
        if hasattr(torchvision.models, "ResNet18_Weights"):
            func1()
        else:
            print("Seems like weights are not implemented, are you using a old pytorch version?")
            func2()
    except BaseException as e:
        def format_exception(e):
            import sys
            import traceback

            exception_list = traceback.format_stack()
            exception_list = exception_list[:-2]
            exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
            exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

            exception_str = "Traceback (most recent call last):\n"
            exception_str += "".join(exception_list)
            # Removing the last \n
            exception_str = exception_str[:-1]

            return exception_str
        print("Error happened during registration:",format_exception(e))
        exit(-1)
