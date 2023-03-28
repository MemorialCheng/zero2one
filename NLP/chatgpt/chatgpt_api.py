# -*- coding: utf-8 -*-
"""
@File : chatgpt_api.py
@Author : cheng
@Date : 2023/3/27
@Description : 

"""

import flask, json
import openai
import os

api = flask.Flask(__name__)
openai.api_key = "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"


@api.route('/chatgpt', methods=['get'])
def chatcompletion():
    prompt = flask.request.args.get("prompt")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="text-davinci-001",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # print(response)
    answer = response["choices"][0]["message"]["content"].strip()
    # print('answer=' + answer)
    # ren = "{'msg': 'success', 'code': 200,'data':'%s'}" % answer
    # return json.dumps(ren, ensure_ascii=False)
    result = prompt + '\n\n' + str(answer)
    return result


if __name__ == '__main__':
    # os.environ["HTTP_PROXY"] = "127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "127.0.0.1:7890"

    # port=80, debug=True, host='0.0.0.0'
    # 如果是0.0.0.0，则可以被外网访问
    api.run()
    # http://127.0.0.1:5000/chatgpt?prompt=我是一名程序员，帮我写一段自我介绍

    # api.run(port=80, debug=True, host='127.0.0.1')
