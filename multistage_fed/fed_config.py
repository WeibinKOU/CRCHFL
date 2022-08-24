#you can manually add more edge server and vehicles repectively by appending items
config = {
        'Edge0' :
        {
            'Vehicle0' :
            {
                'dataset' : './dataset/edge0/vehicle0/',
                'action' : './dataset/edge0/vehicle0/cla7_action.npy',
            },

            'Vehicle1' :
            {
                'dataset' : './dataset/edge0/vehicle1/',
                'action' : './dataset/edge0/vehicle1/cla7_action.npy',
            }
        },

        'Edge1':
        {
            'Vehicle0' :
            {
                'dataset' : './dataset/edge1/vehicle0/',
                'action' : './dataset/edge1/vehicle0/cla7_action.npy',
            },

            'Vehicle1' :
            {
                'dataset' : './dataset/edge1/vehicle1/',
                'action' : './dataset/edge1/vehicle1/cla7_action.npy',
            },

            'Vehicle2' :
            {
                'dataset' : './dataset/edge1/vehicle2/',
                'action' : './dataset/edge1/vehicle2/cla7_action.npy',
            }
        },

        'test':
        {
            'dataset' : './dataset/test/',
            'action' : './dataset/test/cla7_action.npy',
        }
}
