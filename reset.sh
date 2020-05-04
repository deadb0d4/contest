reset() {
  cp blank.cpp main.cpp
  
  rm *.txt
  true || rm bin
  
  touch 0.txt
}

while true; do
    read -p "Do you want to reset? [y/n] " yn
    case $yn in
      [Yy]* ) reset; break;;
      [Nn]* ) break;;
      * ) echo "Please answer [y]es or [n]o...";;
    esac
done
