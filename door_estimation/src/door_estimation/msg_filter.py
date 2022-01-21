import rospy
import queue
import tf 
from functools import reduce

from message_filters import SimpleFilter

class TfMessageFilter(SimpleFilter):
    """Stores a list of messages unless corresponding transforms are 
    available
    """
    def __init__(self, input_filter, base_frame, target_frame,
                 queue_size=500):
        SimpleFilter.__init__(self)
        self.connectInput(input_filter)
        self.base_frame = base_frame
        self.target_frame = target_frame
        # TODO: Use a better data structure
        self.message_queue = queue.Queue(maxsize=queue_size)
        self.listener = tf.TransformListener()
        self.max_queue_size = queue_size
        self._max_queue_size_so_far = 0

    def connectInput(self, input_filter):
        self.queues = [{} for f in input_filter]
        self.incoming_connection = [
            f.registerCallback(self.add, q, i_q)
            for i_q, (f, q) in enumerate(zip(input_filter, self.queues))]
        self.incoming_connection_tf = \
                input_filter[0].registerCallback(self.input_callback)

    def add(self, msg, my_queue, my_queue_index=None):
        my_queue[msg.header.stamp] = msg
        while len(my_queue) > self.max_queue_size:
            del my_queue[min(my_queue)]
        # common is the set of timestamps that occur in all queues
        common = reduce(set.intersection, [set(q) for q in self.queues])
        for t in sorted(common):
            # msgs is list of msgs (one from each queue) with stamp t
            self.msgs = [q[t] for q in self.queues]
            for q in self.queues:
                del q[t]

    def poll_transforms(self, latest_msg_tstamp):
        """
        Poll transforms corresponding to all messages. If found throw older
        messages than the timestamp of transform just found
        and if not found keep all the messages.
        """
        # Check all the messages for transform availability
        tmp_queue = queue.Queue(self.max_queue_size)
        first_iter = True
        # Loop from old to new
        while not self.message_queue.empty():
            msg = self.message_queue.get()
            tstamp = msg.header.stamp
            if (first_iter and 
                self.message_queue.qsize() > self._max_queue_size_so_far):
                first_iter = False
                self._max_queue_size_so_far = self.message_queue.qsize()
                rospy.logdebug("Queue(%d) time range: %f - %f" %
                              (self.message_queue.qsize(), 
                               tstamp.secs, latest_msg_tstamp.secs))
                # rospy.loginfo("Maximum queue size used:%d" %
                #               self._max_queue_size_so_far)
            if self.listener.canTransform(self.base_frame, self.target_frame,
                                          tstamp):
                (trans, quat) = self.listener.lookupTransform(self.base_frame,
                                              self.target_frame, tstamp)
                self.signalMessage(*(self.msgs), (trans, quat))
                # Note that we are deliberately throwing away the messages
                # older than transform we just received
                return
            else:
                # if we don't find any transform we will have to recycle all the messages
                tmp_queue.put(msg)
        self.message_queue = tmp_queue

    def input_callback(self, msg):
        """ 
        Handles incoming message 
        """
        if self.message_queue.full():
            # throw away the oldest message
            rospy.logwarn("Queue too small. If you receive this message too often"
                          + " consider increasing queue_size")
            self.message_queue.get()

        self.message_queue.put(msg)
        # This can be part of another timer thread
        # TODO: call this only when a new/changed transform
        self.poll_transforms(msg.header.stamp)



