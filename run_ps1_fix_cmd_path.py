import os
import argparse

def readFile(pathToFile, headerString="#"):
    """Read file line-by-line  as two lists,
    one containing the header lines,
    the other the content lines

    Parameters:
    -----------
    pathToFile: `str`
        absolute path to file

    Returns:
    ----------
    `list`, `list`
        lists containing the header lines
        and the content lines
    """

    with open(pathToFile) as f:
        allLines = f.readlines()

    headerLines = []
    contentLines = []

    # store the header part and content separately
    for line in allLines:
        if line.startswith("#"):
            headerLines.append(line)
        else:
            contentLines.append(line)

    return headerLines, contentLines


def update_cmd_file_paths(
    bkgnd, pert, root_dir, cmd_file=None
):

    if cmd_file == None:
        cmd_file = f"{bkgnd}BkgndPert{pert}.cmd"

    path_to_cmd = os.path.join(root_dir, cmd_file)

    surfacemap_dir = os.path.join(f"imgCloseLoop_0-{pert}", "iter0", "pert")

    header, content = readFile(path_to_cmd)

    new_content = content.copy()

    for i in range(len(content)):
    
        line = content[i]
        newline = line

        if line.startswith("surfacemap"):

            print(line)
            # split the original line
            splitline = line.split()
            mirror_file = splitline[2].split("/")[-1]
            path_to_mirror = os.path.join(root_dir, surfacemap_dir, mirror_file)
            print(path_to_mirror)
            if not os.path.exists(path_to_mirror):
                raise RuntimeError(f"Error: this file {path_to_mirror} does not exist! ")
     
            # replace that with a new line with an
            # updated path
            newSplitLine = splitline[:2]  # "eg. surfacemap 0"
            # eg. /project/scichris/aos/AOS/DM-28360/imgCloseLoop_0-00/iter0/pert/M1res.txt
            newSplitLine.append(path_to_mirror)
            newSplitLine.append(splitline[-1])  # eg. 1
            newline = " ".join(newSplitLine)
            newline += " \n"
            print("-->", newline)
        new_content[i] = newline
    return new_content


def write_to_file(out_file, content):
    with open(out_file, "w") as output:
        for line in content:
            output.write(line)


def main(root_dir, suffix, bkgnd, perts):

    for pert in perts:
        
        new_content = update_cmd_file_paths(bkgnd=bkgnd, pert=pert, root_dir=root_dir)

        new_cmd_file = f"{bkgnd}BkgndPert{pert}_{suffix}.cmd"
        out_file = os.path.join(root_dir, new_cmd_file)

        print(f"Updated surfacemap paths to {new_cmd_file}, in {root_dir}")
        write_to_file(out_file, new_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update paths to surfacemaps for mirror perturbations in cmd files."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/project/scichris/aos/AOS/DM-28360",
        help="Absolute path to the work directory where .cmd file can be found",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="NCSA",
        help="Suffix added to the cmd file. (default: NCSA)",
    )
    parser.add_argument(
        "--bkgnd",
        type=str,
        default="qck",
        help="Beginning of cmd filename, 'qck' for quickbackground\
        or 'no' for no background. (default: qck)",
    )
    parser.add_argument(
        "--perts",
        nargs="+",
        default=["00"],
        help="A list of perturbations, eg, 00 05. (default: [00]). ",
    )

    args = parser.parse_args()
    main(root_dir=args.root_dir, suffix=args.suffix, bkgnd=args.bkgnd, perts=args.perts)
