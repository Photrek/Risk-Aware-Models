#----------------------------------------------------------------------
# Create random number; sample once and store in file.
#----------------------------------------------------------------------
if [ -e random_container.sh ]
then
    echo loaded random_container.sh
    . ./random_container.sh
    echo value of random_container: $random_container
else
    echo random_container=$RANDOM > random_container.sh
    . ./random_container.sh
    echo just created/executed random_container.sh
fi
#----------------------------------------------------------------------
# Specify image and container name.
#----------------------------------------------------------------------
imgName=srw-vaernn
containerName=cntr-$imgName
