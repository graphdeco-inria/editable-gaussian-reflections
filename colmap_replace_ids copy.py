import sys 

new_ids_path = sys.argv[1]
images_txt_path = sys.argv[2]

print(sys.argv)

new_ids = open(new_ids_path, "r").read().splitlines()

header_lines = []
new_lines = []

with open(images_txt_path, "r") as file:
    for i, line in enumerate(file.readlines()):
        if i < 4:
            header_lines.append(line)
        elif i % 2 == 0:
            fields = line.split(" ")
            fields[0] = new_ids.pop(0)
            new_lines.append(" ".join(fields))

with open(images_txt_path, "w") as new_file:
    for line in header_lines:
        print(line, file=new_file, end="")
    for line in new_lines:
        print(line, file=new_file)
