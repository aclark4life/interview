PROJECT_NAME := interview

e:
	$(EDITOR) programming/interview.py

interview:
	@python programming/interview.py -f 5
	@python programming/interview.py --fibonacci 7
	@python programming/interview.py --search
	@python programming/interview.py --sort
	@python programming/interview.py --queue
	@python programming/interview.py --list
	@python programming/interview.py --tree
	@python programming/interview.py --graph


i:
	$(MAKE) interview
