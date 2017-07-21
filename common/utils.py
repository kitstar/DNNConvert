from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def load_protobuf_from_file(container, filename):
#    if not file_io.file_exists(filename):
#        raise IOError("File %s does not exist." % filename)

    # First try to read it as a binary file.
    with open(filename, 'rb') as fin:
        file_content = fin.read()

    try:
        container.ParseFromString(file_content)
        print("Parse file [%s] with binary format successfully." % (filename))
        return container
    except Exception as e:  # pylint: disable=broad-except
        print("Trying to parse file [%s] with binary format but failed with error %s." % (filename, str(e)))

    # Next try to read it as a text file.
    try:
        from google.protobuf import text_format    
        text_format.Parse(file_content.decode('UTF-8'), container, allow_unknown_extension=True)
        print("Parse file [%s] with text format successfully." % (filename))
    except text_format.ParseError as e:
        raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

    return container


def listToStr(data):
    ret = ""
    first = True
    for e in data:
        if first == False:
            ret += ", "
        ret += str(e)
        first = False
    return ret


