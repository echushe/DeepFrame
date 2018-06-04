#!/usr/bin/env make
# Author: Cong Hai Nguyen
# Created: 2018-05-26


IS_DEBUG:=dnn
ft_exe:=../x64/Debug/facetrain.exe $(IS_DEBUG)
OUT_FILE:=Debug.log


ifneq ($(IS_DEBUG),)
$(shell printf "" > $(OUT_FILE))
endif


.PHONY: all shades face75 face100 facenone pose hid


all:
	$(MAKE) shades

# 2.1.3:
# Train a network using the default learning parameter settings (learning rate 0.3, momentum 0.3) for 75 epochs
shades:
	$(ft_exe) -n shades.net -t straightrnd_train.list -1 straightrnd_test1.list -2 straightrnd_test2.list -e 75 | tee --append $(OUT_FILE)

# /* 2.1.4: answer in the report */
# /* 2.1.5: implement a 1-of-20 face recognizer: accepts an image as input, and outputs the userid of the person (must now be able to distinguish among 20 people)

# (Hint: leave learning rate and momentum at 0.3, and use 20 hidden units)
face75:
	$(ft_exe) -n face.net -t straightrnd_train.list -1 straightrnd_test1.list -2 straightrnd_test2.list -e 75

# /* 2.1.6 */
face100:
	$(ft_exe) -n face.net -t straighteven_train.list -1 straighteven_test1.list -2 straighteven_test2.list -e 100

# /* 2.1.7: report question, no coding */

# /* 2.1.8: take a closer look at which images the net may have failed to classify */
facenone:
	$(ft_exe) -n face.net -T -1 straighteven_test1.list -2 straighteven_test2.list

# /* 2.1.9-2.1.10: Implement a pose recognizer */
pose:
	$(ft_exe) -n pose.net -t all_train.list -1 all_test1.list -2 all_test2.list -e 100

# /* 2.1.11: no coding */

# /* 2.1.12: visualize the weights of hidden unit n */
hid:
	hidtopgm pose.net image-filename 32 30 n

# /* If the images just look like noise, try retraining using facetrain init0 (compile with make facetrain init0), which initializes the hidden unit weights of a new network to zero, rather than random values */

# /* 2.1.13: no coding, just a report question */
