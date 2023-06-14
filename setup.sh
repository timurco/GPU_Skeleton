#!/bin/bash

# Original and new project names
OLD_PROJECT_NAME="GPU_Skeleton"
NEW_PROJECT_NAME="NewProjectName"

# Function for finding and replacing strings
find_and_replace() {
  find . -type f ! -path './.git/*' ! -name '*.sh' -exec sed -i '' -e "s/$1/$2/g" {} +
}

# Export LC_CTYPE to avoid errors with invalid byte sequences
export LC_CTYPE=C
export LANG=C

# Remove existing Git repository
rm -rf .git

# Renaming files and folders
# find . -name "*$OLD_PROJECT_NAME*" ! -path './.git/*' | while read file; do
#   new_file=$(echo "$file" | sed "s/$OLD_PROJECT_NAME/$NEW_PROJECT_NAME/g")
#   mkdir -p "$(dirname "$new_file")"
#   mv "$file" "$new_file"
# done

# Renaming files and folders
find . -name "*$OLD_PROJECT_NAME*" ! -path './.git/*' | while read file; do
  new_file=$(echo "$file" | sed "s/$OLD_PROJECT_NAME/$NEW_PROJECT_NAME/g")
  mv "$file" "$new_file"
done

# Creating new directories based on the original names
find . -type d -name "*$OLD_PROJECT_NAME*" ! -path './.git/*' | while read dir; do
  new_dir=$(echo "$dir" | sed "s/$OLD_PROJECT_NAME/$NEW_PROJECT_NAME/g")
  mkdir -p "$new_dir"
done

# Renaming files and folders
find . -name "*$OLD_PROJECT_NAME*" ! -path './.git/*' | while read file; do
  new_file=$(echo "$file" | sed "s/$OLD_PROJECT_NAME/$NEW_PROJECT_NAME/g")
  mv "$file" "$new_file"
done

# Replace text in files
find_and_replace "$OLD_PROJECT_NAME" "$NEW_PROJECT_NAME"
find_and_replace "$(echo "$OLD_PROJECT_NAME" | tr '[:upper:]' '[:lower:]')" "$(echo "$NEW_PROJECT_NAME" | tr '[:upper:]' '[:lower:]')"
find_and_replace "$(echo "$OLD_PROJECT_NAME" | tr '[:lower:]' '[:upper:]')" "$(echo "$NEW_PROJECT_NAME" | tr '[:lower:]' '[:upper:]')"

# Create a new Git repository
git init
git add --all
git commit -m "Renamed project from $OLD_PROJECT_NAME to $NEW_PROJECT_NAME"
