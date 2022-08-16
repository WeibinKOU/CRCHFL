#you can manually add more edge server and vehicles repectively by appending items
config = {
        'edge0' :
        {
            'vehicle0' :
            {
                'steer_data' : './dataset/edge0/vehicle0/steer/',
                'steer_action' : './dataset/edge0/vehicle0/steer/cla7_action.npy',
                'thro_brake_data' : './dataset/edge0/vehicle0/thro_brake/images/',
                'thro_brake_action' : './dataset/edge0/vehicle0/thro_brake/action.npy'
            },
            'vehicle1' :
            {
                'steer_data' : './dataset/edge0/vehicle1/steer/',
                'steer_action' : './dataset/edge0/vehicle1/steer/cla7_action.npy',
                'thro_brake_data' : './dataset/edge0/vehicle1/thro_brake/images/',
                'thro_brake_action' : './dataset/edge0/vehicle1/thro_brake/action.npy'
            }
        },

        'edge1':
        {
            'vehicle0' :
            {
                'steer_data' : './dataset/edge1/vehicle0/steer/',
                'steer_action' : './dataset/edge1/vehicle0/steer/cla7_action.npy',
                'thro_brake_data' : './dataset/edge1/vehicle0/thro_brake/images/',
                'thro_brake_action' : './dataset/edge1/vehicle0/thro_brake/action.npy'
            },

            'vehicle1' :
            {
                'steer_data' : './dataset/edge1/vehicle1/steer/',
                'steer_action' : './dataset/edge1/vehicle1/steer/cla7_action.npy',
                'thro_brake_data' : './dataset/edge1/vehicle1/thro_brake/images/',
                'thro_brake_action' : './dataset/edge1/vehicle1/thro_brake/action.npy'
            }
        }
}