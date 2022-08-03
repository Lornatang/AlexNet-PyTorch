import argparse
import json
import traceback

from tiyaro.sdk.test_utils.util import (TEST_COMMAND_ARG_INPUT,
                                        TEST_COMMAND_ARG_OUTPUT,
                                        TEST_COMMAND_ARG_PRETRAINED,
                                        TEST_COMMAND_ARG_VERBOSE,
                                        get_input_by_model_type,
                                        get_pretrained_file_path,
                                        validate_and_save_test_input,
                                        validate_and_save_test_output)

from tiyaro_handler.model_handler import TiyaroHandler

if __name__ == "__main__":
    try:
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument(
            TEST_COMMAND_ARG_PRETRAINED, type=str, required=False)
        my_parser.add_argument(TEST_COMMAND_ARG_INPUT, type=str, required=True)
        my_parser.add_argument(TEST_COMMAND_ARG_OUTPUT,
                               type=str, required=False)
        my_parser.add_argument(TEST_COMMAND_ARG_VERBOSE,
                               type=str, required=False)
        args = my_parser.parse_args()

        is_verbose = args.verbose
        if is_verbose == "True":
            is_verbose = True
        else:
            is_verbose = False

        file_path = get_pretrained_file_path(args.pretrained, is_verbose)
        test_input = get_input_by_model_type(args.input, is_verbose)
        output_file = args.output
        if output_file == "None":
            output_file = None

        ob = TiyaroHandler()
        ob.setup_model(pretrained_file_path=file_path)
        ob.declare_schema()
        print(f"INIT - Done")

        validate_and_save_test_input(ob, test_input, is_verbose)

        print(f"INFERENCE - Started")
        test_output = ob.infer(test_input)
        print(f"INFERENCE - Done")

        validate_and_save_test_output(ob, test_output, is_verbose)

        print("OUTPUT STARTS - {}".format('*' * 50))
        print(json.dumps(test_output, indent=4, sort_keys=True))
        print("OUTPUT ENDS - {}".format('*' * 50))

        if output_file:
            if not ".json" in output_file:
                output_file += ".json"
            with open(output_file, "w+", encoding="utf-8") as f:
                json.dump(test_output, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"ERROR - {e}")
        traceback.print_exc()
        exit(-1)
