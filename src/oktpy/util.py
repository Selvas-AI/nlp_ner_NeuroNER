import gc
import os

import jpype


def init_jvm():
    max_memory = 400
    jvm_path = jpype.get_default_jvm_path()
    file_path = os.path.dirname(os.path.realpath(__file__))
    jars_path = os.path.join(file_path, 'jars')
    jars = [os.path.join(jars_path, f) for f in os.listdir(jars_path)]
    jpype.startJVM(jvm_path, '-Djava.class.path={}'.format(os.pathsep.join(jars)), '-Dfile.encoding=UTF8', '-ea',
                   '-Xmx{}m'.format(max_memory))


def load_okt():
    if not jpype.isJVMStarted():
        init_jvm()

    return jpype.JPackage('net.ingtra.pyokt').OpenKoreanTextWrapper


call_count = 0


def attachThread(original_function):
    def wrapper_function(*args, **kwargs):
        global call_count
        call_count += 1
        if call_count > 1000:
            gc.collect()
            call_count = 0
        if jpype.isJVMStarted():
            jpype.attachThreadToJVM()
        return original_function(*args, **kwargs)

    return wrapper_function


def java_bool(bool):
    return jpype.java.lang.Boolean(bool)
