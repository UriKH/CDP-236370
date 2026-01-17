#
#   @date:  [19/03/2024]
#   @author: [Michal Ozeri, Guy Sudai]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing

class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.reader_side, self.writer_side = multiprocessing.Pipe(False)
        self.writers_lock = multiprocessing.Lock()

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.writers_lock.acquire()
        self.writer_side.send(msg)
        self.writers_lock.release()


    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        return self.reader_side.recv()
    
    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''
        return self.reader_side.poll()
